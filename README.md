English | [简体中文](https://github.com/Alexw1111/laya-cuda/blob/main/README.zh-CN.md)

# laya-cuda

A lightweight inference library for [Laya](https://github.com/NandhaKishorM/laya)

![](https://raw.githubusercontent.com/Alexw1111/laya-cuda/main/docs/assets/benchmark.png)

Latency of one request repeated 200 times:

| Model | Request | laya-cuda P50 / P95 | Official SDK P50 / P95 | Speedup P50 / P95 |
|---|---|---:|---:|---:|
| `laya` | short, 83 tokens | 2.3 / 2.9 ms | 19.3 / 20.5 ms | 8.3× / 7.2× |
| `laya` | full context, 512 tokens | 5.6 / 6.0 ms | 18.9 / 20.0 ms | 3.4× / 3.3× |
| `laya-multilingual` | short, 81 tokens | 1.5 / 2.0 ms | 16.1 / 17.0 ms | 10.6× / 8.4× |
| `laya-multilingual` | full context, 1,024 tokens | 4.9 / 5.4 ms | 15.8 / 17.0 ms | 3.2× / 3.1× |
| `laya-typed-decisions` | short, 83 tokens | 2.4 / 2.9 ms | 19.6 / 20.8 ms | 8.3× / 7.1× |
| `laya-typed-decisions` | full context, 1,024 tokens | 9.6 / 9.9 ms | 19.8 / 21.3 ms | 2.1× / 2.2× |

Accuracy on 200 frozen requests (laya-cuda / SDK): `laya` 52.25% / 52.25%, `laya-multilingual` 43.25% / 43.25%, `laya-typed-decisions` 69.25% / 69.75%. Jev, a larger remote model, scores 65.00% on the `laya` requests at 378 / 466 ms, which includes network latency.

- **Setup:** RTX 4090, Windows 11. The short request is a real AG News request with one four-option question; the full-context request asks the same question about real text truncated at the model's limit. Each request ran 200 times per process, in 3 processes per backend with alternating order.
- **What is timed:** the full `predict` call, including tokenization, GPU work and decoding. Jev's time also includes the network.
- **Mixed requests:** on 200 different frozen requests (40–900 tokens, 1–5 questions, 4–77 options, each new shape's first request included), `laya` measures 4.1 / 8.5 ms against 20.1 / 22.1 ms for the SDK: 4.9× / 2.6×. Mixing request sizes raises P95; the [latency report](https://github.com/Alexw1111/laya-cuda/blob/main/reports/latency/results.md) explains it.
- **SDK precision:** the official SDK runs at its default BF16. On the 2,000-decision fixed set, laya-cuda agrees with the SDK at FP32 more closely than BF16 does.
- **Protocol:** [benchmarks/](https://github.com/Alexw1111/laya-cuda/blob/main/benchmarks/README.md); the [latency report](https://github.com/Alexw1111/laya-cuda/blob/main/reports/latency/results.md) records the frozen inputs.

## Install

```sh
pip install laya-cuda   # core: CuPy + CUDA components, no Torch
laya-cuda doctor laya   # check driver, GPU and CUDA components, then time one prediction
```

Windows or Linux x86-64, Python 3.12–3.14 and an NVIDIA driver; no CUDA Toolkit is needed. To work on the source, use [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/Alexw1111/laya-cuda
cd laya-cuda
uv sync
uv run laya-cuda doctor laya
```

## Use

```python
from laya_cuda import Engine

questions = {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "payments and refunds", "technical": "software errors"}},
    "urgent": {"type": "noul", "instructions": "Does this need urgent attention?"},
}
with Engine("laya") as model:
    print(model.predict("Please refund my duplicate payment.", questions)["answers"])
```

```sh
laya-cuda predict "Please refund my duplicate payment." -q questions.json
laya-cuda predict -i requests.jsonl -q questions.json -o results.jsonl
```

Answers keep the SDK's structure: probabilities, confidence, action probability and token usage. `BatchEngine` batches concurrent callers. `backend="official"` runs the unmodified SDK for comparison and needs the `full` or `reference` extra.

## Learn more

- **[Guide](https://github.com/Alexw1111/laya-cuda/blob/main/docs/guide.md):** `Engine`, `BatchEngine`, model registry, platform and driver details, development, architecture.
- **[Examples](https://github.com/Alexw1111/laya-cuda/blob/main/examples/README.md):** decision workflows, the Jev client and Snake Lab, a live demo where every move is a Laya prediction.
- **[Benchmarks](https://github.com/Alexw1111/laya-cuda/blob/main/benchmarks/README.md):** how comparisons, gates and reports work.
- **[FP32-reference validation](https://github.com/Alexw1111/laya-cuda/blob/main/reports/fp32-reference/results.md):** the full record of the numerical, long-input and accuracy gates.
- **[Latency tail](https://github.com/Alexw1111/laya-cuda/blob/main/reports/latency/results.md):** what made P95 twice P50, the fixes, and their check on untuned workloads.
- **[Adding a model](https://github.com/Alexw1111/laya-cuda/blob/main/docs/extending.md):** the adapter recipe.

The library is about 1,270 lines of Python and CUDA. CuPy owns memory, streams and CUDA Graphs; cuBLAS does the matrix multiplication; a few NVRTC kernels handle the rest, including one fused tensor-core attention kernel.

## Acknowledgements

- First, thanks to [Laya](https://github.com/NandhaKishorM/laya) and its developers. Without Laya, this project would not exist.
- Thanks to the benchmarks: [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [DAIR Emotion](https://huggingface.co/datasets/dair-ai/emotion), [Banking77](https://huggingface.co/datasets/mteb/banking77) and [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent).
- Thanks to [laya-mlx](https://github.com/mizorewww/laya-mlx), which inspired this project.

## License

Source code and derived request semantics are Apache-2.0; see [LICENSE](https://github.com/Alexw1111/laya-cuda/blob/main/LICENSE) and [laya_cuda/NOTICE](https://github.com/Alexw1111/laya-cuda/blob/main/laya_cuda/NOTICE). Model and dataset licenses apply separately.
