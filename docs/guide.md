# laya-cuda guide

## API

### `Engine`

`Engine(model, *, revision=None, device="cuda:0", backend="cuda", max_cached_shapes=16, registry=None)`

- **`model`:** a registered alias or a local checkpoint directory. Aliases pin immutable Hugging Face revisions, and `revision` overrides the pin. A local directory uses its existing files and takes precedence over aliases.
- **`backend="cuda"`:** FP16 with FP32 accumulation and normalization. Attention runs on tensor cores with an FP32 softmax; probabilities enter the value product as FP16 high and low parts, which keeps FP32-level precision. Decision logits stay FP32. Accuracy is validated against the SDK at FP32; see [the FP32-reference report](../reports/fp32-reference/results.md).
- **`backend="official"`:** calls the unmodified SDK at its default checkpoint precision.
  - It requires the `reference` extra and a CUDA-enabled Torch.
  - The adapter checks CUDA availability and the devices of the model tensors and forward inputs, and rejects CPU fallback.
  - The core wheel environment deliberately has no Torch, so it is not the environment used to time the official baseline.
- **No silent switching:** backend selection is manual; there is never a silent fallback to another backend.
- **Shape classes:** a request's questions are packed back to back, each starting at a multiple of 16 tokens, so no work is spent on padding to the longest question. Each shape class (packed tokens rounded up in 32-, 64- or 128-token steps; question and option counts rounded up to powers of two) runs as one CUDA graph. The first request of a class captures that graph, which costs a few milliseconds once; `metrics["cache_miss"]` marks it. Up to `max_cached_shapes` classes (default 16, at most 64) stay resident, least recently used first out.
- **Lifecycle:** engines are synchronous and not thread-safe. Release them with `close()` or a `with` block. Complete requests are not cached.
- **Inputs:** state can be text or JSON-compatible data.

### Model registry

Built-in aliases are data in [`laya_cuda/models.json`](../laya_cuda/models.json). Pass an external JSON file to add or override aliases without modifying the installed package:

```json
{
  "my-laya": {"path": "./weights/laya"},
  "pinned-laya": {
    "repo_id": "convaiinnovations/laya",
    "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
  }
}
```

```python
with Engine("my-laya", registry="models.json") as model:
    result = model.predict("Please refund my payment.", questions)
```

- **Entry format:** each entry has either `path`, or both `repo_id` and `revision`. Relative paths resolve against the JSON file's directory.
- **Overrides:** external entries override matching built-ins, and the other built-ins remain available. `BatchEngine` accepts the same `registry` option.
- **No new architectures:** a registry entry does not enable a new architecture; every checkpoint still passes the adapter's compatibility checks.

**Fine-tuned checkpoints:** if the Laya architecture and export format are unchanged, point `Engine` at the new directory or add an alias; no Python changes are needed. Supply the complete weights, encoder/task configuration and tokenizer. See the [integration guide](extending.md) for the export layout and comparison command. Unmerged LoRA adapters and different task heads need additional work.

### `BatchEngine` (concurrent callers)

An opt-in, CUDA-only wrapper built on the standard library. One worker owns one engine and combines complete requests. Questions are packed back to back, so requests of different lengths share a batch.

```python
from laya_cuda import BatchEngine

with BatchEngine("laya") as model:
    first = model.submit("Please refund my payment.", questions)
    second = model.submit("The application crashes.", questions)
    print(first.result()["answers"], second.result()["answers"])
    print(first.metrics)
```

- **Limits:** defaults are 16 questions and 4,096 packed tokens per batch (each question rounded up to a multiple of 16), 128 pending requests (including those executing), and up to 1 ms of collection delay. Tune them with `max_batch_size`, `max_batch_tokens`, `max_pending` and `max_wait_ms`. For example, ten 1,024-token questions need `max_batch_tokens=10240` or more.
- **Oversized requests:** an oversized request fails through its Future. It is never truncated or split to fit a scheduling budget; checkpoint truncation semantics still apply.
- **Concurrency:** `submit` snapshots its inputs and is safe for concurrent callers. A full queue raises `queue.Full` immediately. `future.cancel()` succeeds only before the worker starts preparing that request.
- **Shutdown:** `close()` rejects new work, drains accepted requests and releases the engine.
- **Callbacks:** do not block inside Future callbacks waiting on the same engine, because callbacks can run on the GPU worker.
- **Metrics:** `future.metrics` reports request latency and queue/preparation time. GPU time, cache misses and allocator bytes describe the **shared batch**, not the individual request.
- **Scheduling:**
  - Collection delay does not bound total queue latency, and a batch that is already running cannot be preempted.
  - Batches fill in arrival order. A request that does not fit waits, and the oldest waiting request always starts the next batch, so none is starved indefinitely.
- **When not to use it:** use `Engine` for isolated latency-sensitive calls, because batching adds latency at low concurrency.

## Platform and dependencies

- **Targets:** Windows x86-64 and Ubuntu WSL2 x86-64, Python 3.12–3.14, NVIDIA GPU with compute capability 8.0 or newer. Model validation used Python 3.12 on an RTX 4090; installation was also checked with Python 3.13 and 3.14 on Windows. Older drivers, other GPUs and native Linux are not validated.
- **GPU generation:** the attention kernel uses FP16 tensor-core `mma.sync.m16n8k16`, which needs Ampere (sm_80) or later. Turing GPUs such as the T4 and RTX 20 series, and older ones, cannot compile it: `Engine` raises a clear error and `laya-cuda doctor` flags them.
- **Drivers:** Windows 551.78+ (including the host driver used by WSL2) or Linux 550.54.15+; newer drivers remain compatible. These targets follow [NVIDIA's CUDA 12.4.1 release notes](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/).
  - The lower CUDA 12 minor-compatibility floor is not advertised for our runtime compilation path.
  - The older targets have not been physically tested.
- **Core wheels:** `cupy-cuda12x` 14 and the CUDA 12 runtime, cuBLAS and NVRTC wheels, 12.4 or newer.
  - No CUDA Toolkit, host compiler, NVCC, manual library path, model conversion or CUDA 13 is required.
  - The dependencies still occupy hundreds of megabytes; this is not a tiny pure-Python runtime.
- **Version ranges:**
  - Each dependency accepts releases up to its next major version, so the package can share an environment with other CUDA 12 packages. A fresh install currently resolves the CUDA components to 12.9; `uv.lock` keeps the 12.4 set that the benchmarks used.
  - Checked on the RTX 4090: with CUDA components 12.9 the quality gates give the same results as with 12.4, and fresh installs pass on Windows (Python 3.12–3.14) and WSL2 (Python 3.12). At the lower bounds (CUDA components 12.4, numpy 2.0, tokenizers 0.23.1, huggingface-hub 1.5) answers and token IDs match the current versions exactly.
  - The ranges do not pin your system Toolkit or driver. CuPy 14.2 reports an embedded runtime of 12.9; runtime version strings alone are not driver requirements.
- **Not supported:** CUDA 11. Also, do not install `cupy-cuda12x` and `cupy-cuda13x` in the same environment.
- **Extras:**
  - `reference` adds the official Laya SDK. It requires CUDA 13 Torch, which has a higher driver requirement; this does not apply to core-only installs.
  - `benchmark` adds reference and measurement/data tools. `full` is the union of both.
  - Extras never change the default backend or import Torch eagerly.
- **Official backend on Windows:** PyPI only publishes a CPU Torch for Windows, so add PyTorch's CUDA index when installing the extras: `pip install "laya-cuda[full]" --extra-index-url https://download.pytorch.org/whl/cu130`. The pinned `torch==2.12.1` then resolves to `2.12.1+cu130`. With uv, `uv sync --extra full` selects that index automatically. Linux wheels from PyPI already include CUDA. Core CUDA inference does not depend on Torch.

## Development

```sh
uv sync --locked --extra full --group dev
uv run --extra full --group dev pytest -q
uv run --extra benchmark python -m benchmarks.run --models laya --models-dir models \
  --dataset benchmarks/data/typed-decisions.parquet --output reports/benchmark
uv build
```

- **Model tests:** tests and benchmarks use local checkpoints in `models/laya`, `models/laya-multilingual` and `models/laya-typed-decisions`; model tests are skipped when they are absent. `uv sync --locked` also replaces an older CuPy distribution in an existing checkout.
- **CI:** GitHub Actions ([tests.yml](../.github/workflows/tests.yml)) runs the tests that need no GPU, `pytest -m "not gpu and not reference"`, on Ubuntu and Windows for every push to `main` and every pull request. Hosted runners have no GPU, so run the full suite locally before merging changes to kernels, the runtime or numerics.
- **Benchmark protocol:** 20 warmups, 200 samples and three rounds, with backend order alternating across isolated processes. Run with the GPU otherwise idle and use a fresh output directory each time.
- **What latency includes:** API latency covers tokenization, transfers, execution and decoding. Device-event timing is reported separately where available.
- **Short runs:** short diagnostic runs cannot produce verified-speedup flags.
- **Evidence:**
  - [FP32-reference validation](../reports/fp32-reference/results.md): the current numerical, long-input and accuracy gates on Windows and WSL2.
  - [Latency tail](../reports/latency/results.md): the diagnosis and fixes behind the current P95, checked on untuned workloads.
  - [Comparison guide](../benchmarks/README.md): frozen upstream datasets and the pinned upstream SDK. See [dataset provenance](../benchmarks/data/README.md); pseudo-labels are not human-verified ground truth.

## Architecture

```text
laya_cuda/    Installable library: engine.py (public lifecycle), batching.py (BatchEngine), cli.py (laya-cuda command),
              runtime.py / ops.py / kernels.cu (resident GPU execution, packed shape classes, fused attention),
              laya.py (checkpoint and task semantics), models.py + models.json (pinned aliases, adapter selection)
examples/     Small programs using the public API, including Snake Lab
benchmarks/   Official comparison harness and evaluation helpers
tools/        Profiling, diagnostics, comparison and installation utilities
tests/        Correctness and lifecycle tests
docs/         Extension documentation
reports/      Recorded experiment results, outside distribution archives
```

- **What gets packaged:** only `laya_cuda/` goes into the wheel. The source distribution adds examples, benchmarks, tests, tools and docs. Neither artifact bundles model weights, environments, generated reports or build caches.
- **Adding a model:** new models follow [the extension recipe](extending.md). Compatibility must be tested; arbitrary ModernBERT support is not advertised.
- **Previous implementation:** the previous working tree is kept read-only under `old/`, including archive hashes and Git state snapshots. The library does not import it, and its historical performance numbers are not current evidence.
