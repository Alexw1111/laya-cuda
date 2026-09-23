# Benchmarks

## Charts and reproducible task comparisons

`benchmarks.compare` now writes a Markdown `report.md`, exportable PNG/SVG
charts, per-round `metrics.csv`, raw JSON, and a `reproduce/` source-and-input
bundle automatically. Plotting dependencies are confined to the benchmark extra.
It also writes a sibling `.zip` and SHA-256 checksum for sharing, using an explicit
file list so later-created virtual environments, caches and credentials stay out.
The core inference wheel does not import Matplotlib.

```powershell
uv sync --locked --extra benchmark
uv run --no-sync python -m benchmarks.prepare --per-suite 25 --option-permutations --output reports/frozen-inputs.jsonl
uv run --no-sync python -m benchmarks.compare --cases-file reports/frozen-inputs.jsonl --checkpoint laya --rounds 3 --output reports/my-comparison
```

The `upstream` backend needs the pinned checkout described below. A local model
directory is also accepted; add `--checkpoint-source laya` only when it came from
that pinned public alias. On replay this is a download hint, not an exemption
from the full model/configuration/tokenizer hash check.

Open `reports/my-comparison/report.md` in any Markdown viewer, including GitHub.
It links the chart images next to it and needs no web server, CDN or Internet
connection. The PNG/SVG files also support slides and
papers; CSV contains the plotted per-round statistics and quality coverage.
Each bar is a median across rounds, with min–max whiskers, not a confidence
interval. Repeated rounds do not create more independent evaluation examples.
Failures, missing data and small-sample tail-latency limitations remain visible.

The report also includes `cache-diagnostics.csv`: cache hits/misses/unknown,
subset P99 and the GPU event time at the slowest API call. These are diagnostic
subsets, not replacements for end-to-end timing. Warmup repeats the first input
only. New shapes and evicted shapes still pay workspace/graph preparation costs.
For 25 requests the interpolated P99 is dominated by the largest observation;
use a separate fixed-shape, thoroughly warmed repeated workload (for example,
at least 1,000 calls per shape) when studying steady-state tail stability.

Share the generated ZIP (or the output directory before adding local environments),
then the other person extracts it and runs from its
`reproduce/project` directory:

```powershell
uv sync --locked --extra benchmark --python 3.12
uv run --no-sync python -m benchmarks.reproduce --bundle .. --output ../../replayed
```

Use the exact Python patch version in `reproduce/REPRODUCE.md` for the closest
environment match. Install uv and a supported NVIDIA driver first; Git is needed
to fetch the pinned upstream SDK. Platform-dependent wheel variants can differ.
`uv sync --locked` refuses to silently update the dependency lockfile.

For private/local weights without a public source hint, pass
`--checkpoint /path/to/identical/model`. The bundle does not include weights,
virtual environments or API keys. A Jev replay requires `OPENROUTER_API_KEY` and
makes paid requests; remote alias/provider updates can change results.

Replay verifies the frozen input bytes, source snapshot and lockfile before
execution, then verifies **every consumed checkpoint file**, including the
tokenizer. Changed or missing files stop the replay. Each worker runs from the
captured source, even when the current repository has uncommitted changes.
Actual Python/package versions, GPU/driver, precision, warmup count, rounds,
input fingerprints and checkpoint fingerprints are preserved. Completed raw
results receive hashes; incomplete worker matrices or changed results cannot
silently render as a completed comparison.

This reproduces the workload and method, not identical timing on arbitrary
hardware. Use an idle GPU, preserve warmup and precision, and compare environment
records before interpreting differences. First-N dataset selection and option
permutations are fixed in the copied JSONL; replay never downloads a newer
dataset. Public model downloads remain pinned and must match recorded bytes.

Existing task results can be rendered without rerunning inference:

```powershell
uv run --no-sync python -m benchmarks.report reports/my-comparison
uv run --no-sync python -m benchmarks.reproduce --bundle reports/my-comparison/reproduce --verify-only
```

`--verify-only` checks the bundle, not the model or installed environment. Legacy
results without a bundle are visibly marked as having incomplete provenance;
rendering cannot retroactively make an old run reproducible. Add `--note` to the
report command to retain known measurement limitations. The dedicated
`benchmarks.run` latency promotion suite keeps its separate JSON/gate protocol;
this report format currently applies to `benchmarks.compare` task comparisons.

These tools run from the repository root and are not included in the installed
`laya_cuda` package. Install the optional measurement dependencies with
`uv sync --extra benchmark`.

```sh
uv run --extra benchmark python -m benchmarks.run --models laya --models-dir models --dataset benchmarks/data/typed-decisions.parquet --output reports/benchmark-new
```

Custom fine-tunes use the same harness:

```sh
uv run --extra benchmark python -m benchmarks.run --models my-finetune --registry models.json --dataset benchmarks/data/typed-decisions.parquet --output reports/my-finetune
```

With `--registry`, the harness resolves aliases through that JSON for every backend
instead of constructing paths under `--models-dir`. Results and quality gates also
include custom aliases. Use held-out evaluation data relevant to the fine-tune.

The harness uses isolated processes, alternating backend order, and synchronized
prediction results. Defaults remain 20 warmups, 200 samples and three rounds.
Use a fresh output directory. Short smoke runs verify execution only and cannot
promote a speedup. The implementation hashes describe the library actually loaded
by each worker; the harness hash is recorded separately.

Numerical reference and gates:

- **Reference:** the official SDK run at FP32 (`official-fp32`), the checkpoint's
  own precision. It runs once, in round 0, and records answers only.
- **Peers, not ground truth:** SDK FP16 and the SDK default BF16 are reduced-precision
  peers. They are timed and reported beside the reference.
- **Gates:** unchanged thresholds, now against the reference. Probability MAE <= 0.001,
  maximum <= 0.01, the same limits for action probability, and decision agreement >= 99.5%.
- **Accuracy gate:** fixed-set accuracy may not be significantly below the reference. Among
  the decisions exactly one of the two gets right, a one-sided exact sign test must give
  p >= 0.05. Until 2026-09-24 the rule was "no fewer correct decisions than the reference";
  single near-tie decisions flipped that result in both directions between numerically
  equivalent versions, so it was replaced after observing results. This loosens
  the gate: with the agreement gate, up to 6 fewer correct decisions out of 2,000 can pass,
  where the old rule allowed none. The rule is frozen; see the held-out check in
  [the FP32-reference report](../reports/fp32-reference/results.md#held-out-check-of-the-accuracy-gate-pre-registered-2026-09-24).
- **Length gate:** a suite of 48 long inputs built by joining real fixture states,
  spanning 60%–100% of the model's context budget, must pass the same gates.
- **Stress probes:** the repeated-token length probes (`l768-q10`, `boundary-l1023`, …)
  are stress diagnostics. They record the error of laya-cuda and of SDK FP16 against the
  reference, but they do not gate. On `laya-multilingual`, repeated tokens create
  massive activations that amplify any FP16 rounding, including the SDK's own.
- **Why the reference changed:** until 2026-09-23 the reference was SDK FP16, and
  accuracy was compared with BF16. Both deviate from FP32 by more than the gates allow
  on inputs where laya-cuda does not. See [the FP32-reference report](../reports/fp32-reference/results.md).

Other repository tools remain outside the library:

- `tools/benchmark_batching.py`: serial versus concurrent request measurements.
- `tools/compare_optimization.py`: an earlier wheel versus the current library and SDK.
- `tools/profile_long.py` and `tools/compare_layers.py`: profiling and numerical diagnostics.
- `tools/render_report.py`: reports from completed measurements.
- `tools/verify_install.py` and `tools/install_size.py`: installation checks.

`laya-cuda[benchmark]` and `[full]` provide optional dependencies, not an installed
benchmark command. The former `laya-cuda-bench` command and
`laya_cuda.benchmark` module have been replaced by `python -m benchmarks.run`.
Existing reports retain their historical results; this reorganization does not
produce new performance evidence.

## Upstream datasets and Jev comparison

Pin the real upstream library, outside the distributed package:

```powershell
git clone https://github.com/nandhakishorm/laya.git vendor/laya-upstream
git -C vendor/laya-upstream checkout 573e5b62696ba441230cd6be71d593331b5d23af
uv sync --extra benchmark
uv run --no-sync python -m benchmarks.prepare --per-suite 100 --option-permutations --output reports/task-inputs.jsonl
uv run --no-sync python -m benchmarks.compare --cases-file reports/task-inputs.jsonl --backends cuda upstream --rounds 3 --output reports/task-comparison
```

`prepare` supports AG News, DAIR Emotion, Banking77, MASSIVE and local
typed-decisions parquet. It adapts the pinned upstream input builders, records
dataset revisions/fingerprints and freezes one JSONL for all backends. MASSIVE
is opt-in: `--suites massive --languages en zh-CN --per-suite 100`. Choice reversal
adds a separate order-sensitivity suite; ordinal score rubrics are never reversed.
See [upstream attribution and scope](UPSTREAM_NOTICE.md). This is not a claim to
have reproduced every upstream benchmark or its raw-logit/calibration pipeline.

`compare` uses separate processes and alternating backend order. `upstream`
imports the unmodified SDK from the pinned checkout and calls its real public
prediction path through our thin lifecycle wrapper. It uses the same local
weights as `cuda`, while preserving each backend's precision and recording it.
`official` instead uses the separately installed SDK. Published upstream CPU/T4
and third-party Jev results are never inserted into the measured rows.

The official adapter checks not only `torch.cuda.is_available()` but the actual
model parameters/buffers and forward-input tensor devices. CPU fallback aborts
the comparison. New task worker JSON includes `runtime_device` with Torch/CUDA
build versions, GPU name and observed model/input devices. The benchmark uv
environment uses CUDA-enabled Torch; a separate lightweight wheel-test
environment deliberately has no Torch and is not an official baseline.

With `OPENROUTER_API_KEY` set, add `jev` to `--backends`. The remote input is the
same complete state/question payload; checkpoint context/truncation policies
still differ and must be considered when interpreting results. `--jev-model`
can pin a supported dated model ID. Costs include warmup requests; the default
reported-cost stop is USD 1 per worker and each round is a separate worker.
It is not a hard billing cap. There are no hidden retries or silent model fallbacks.

Each worker retains request hashes, answers, errors, usage, P50/P95/P99 inputs,
load/warmup timings, implementation/weight hashes and sampled RSS. Accuracy counts
failed requests as incorrect; latency quantiles describe successful requests and
all attempted latencies remain in the report. Calibration metrics use maximum
label probability rather than the SDK's entropy-derived `confidence`. Missing
distributions are not invented; probabilistic coverage is reported separately.
If a scalar score lacks a class distribution, its MAE is retained but aggregate
class accuracy is unavailable rather than inventing an argmax from the score.
The harness does not fit temperatures on evaluation data or issue verified-speedup
flags. The typed-decisions labels are synthetic, and AG News has training overlap.

Use `--checkpoint models/laya-multilingual` or `models/laya-typed-decisions` to
evaluate those checkpoints explicitly. Their support status follows the
`benchmarks.run` gates (see the main README). Dataset requests or model failures
are not evidence of successful support.

For throughput under concurrent request submission, keep using
`tools/benchmark_batching.py`. For a detailed latency sweep with frozen numerical
promotion gates, keep using `benchmarks.run`. The task-comparison harness above
adds dataset quality and remote deployment comparisons; it does not replace those
specialized measurements.

### Chart styling

`benchmarks/style.py` owns the shared muted palette and chart typography.
Backend colors stay stable across plots. Regenerate `report.md`, PNG and SVG from existing
raw results with `uv run --no-sync python -m benchmarks.report <result-directory>`.
This does not rerun inference or modify frozen worker snapshots. Historical bundles
retain their original renderer; use the current checkout to reproduce the new styling.
