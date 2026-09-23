# Adding a fine-tuned checkpoint

A fine-tune that preserves the supported Laya architecture and checkpoint format
uses the existing adapter and GPU engine. No library code changes or new adapter
are needed. Repository names, directory names and weight values are not allowlisted.

Export the complete checkpoint with this layout:

```text
my-checkpoint/
  model.safetensors
  rl_agent_config.json
  encoder/config.json
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
  tokenizer/special_tokens_map.json
```

Tokenizer metadata must define CLS, SEP, MASK and PAD tokens; those definitions
may be in either tokenizer metadata file. Keep the task configuration, calibration
and tokenizer from the fine-tune alongside its weights. Weights use the same
tensor names and shapes expected from that configuration. Export one complete
`model.safetensors`; adapter-only LoRA files, sharded weights and arbitrary
Transformers export layouts are not currently supported. Merge adapters and export
the complete Laya checkpoint in the training environment before distribution.

Use a local checkpoint directly:

```python
from laya_cuda import Engine

with Engine("./my-checkpoint") as model:
    result = model.predict(state, questions)
```

Or add an alias to your own `models.json`:

```json
{
  "my-finetune": {"path": "./my-checkpoint"}
}
```

```python
with Engine("my-finetune", registry="models.json") as model:
    result = model.predict(state, questions)
```

For a hosted checkpoint, replace `path` with `repo_id` and `revision`; pin the
checkpoint's commit hash for reproducibility. `BatchEngine` accepts the same
configuration. The runtime checks architecture, tensor names and tensor shapes;
it loads the checkpoint's weights rather than substituting a built-in model.
The encoder alone being ModernBERT is insufficient if the decision heads or input
and output semantics differ from Laya.

Validate each fine-tune against the official SDK using that same checkpoint and
representative labeled evaluation data. The external harness accepts custom aliases:

```sh
uv run --extra benchmark python -m benchmarks.run --models my-finetune --registry models.json --dataset benchmarks/data/typed-decisions.parquet --output reports/my-finetune
```

Replace the example dataset with held-out data appropriate to the fine-tune using
the [dataset schema](../benchmarks/data/README.md). All compared backends resolve the same
alias, and the results record checkpoint hashes. Loading successfully establishes
structural compatibility, not task quality or a speedup. The synthetic extension
test verifies changed weights, external aliases, direct paths, batching and rejection
of mismatched tensors; it does not validate a real third-party fine-tune's quality.

## Adding a new model architecture or task contract

Start from the implemented Laya adapter, not a copied GPU engine:

1. Pin the official checkpoint, tokenizer and SDK. Record file hashes and request/response fixtures. Check tensor names, dimensions, normalization, attention, RoPE and task heads.
2. Add one adapter module exposing the configuration fields used by `Runtime`, `weight_shapes()`, `prepare()`, `decode()`, `allocate_head(slot)`, `forward_head(slot)` and `read_outputs(slot)`. Preparation returns token IDs, marker positions and task types. The adapter owns head buffers and decoding; compatible heads use existing slot primitives.
3. Register the constructor once in `models.adapter`, add checkpoint aliases to `laya_cuda/models.json` or supply an external JSON registry, and extend file resolution only if the format requires it. Adding a compatible checkpoint alias requires no Python changes. No runtime import should depend on an official heavyweight SDK.
   Supply `official(device)` to construct that model's thin SDK adapter when the optional official backend is requested. It exposes `predict`, `close`, `precision` and `gpu_ms`; do not embed SDK-specific access in `Engine`.
4. Compare tokens/masks, layer outputs, probabilities and decisions with that model's own official implementation. Add lifecycle and short/long benchmark cases. Admit support only after the frozen gates pass.

The current executor supports bias-free exact-GELU ModernBERT blocks, 64-dimensional attention heads, unscaled RoPE, periodic global attention and symmetric local windows, up to 1024 tokens. Tensor/configuration checks reject unsupported variants. Compatible task heads can use `Slot.norm`, `Slot.linear`, `Slot.attend` and the shared kernels. A different encoder may require executor changes; it is not automatically an adapter-only addition.

Keep adapters explicit. Do not add a speculative second model, universal compiler, plugin loader or automatic backend dispatcher. Input serialization and decoding belong in adapters; allocation, bounded graph slots and resource ownership belong in the runtime.

A slot packs a request's questions back to back into `slot.m` token rows, each
question starting at a multiple of 16 tokens. Per-token inputs (`positions`,
`rows`, `kinds`) and per-question inputs (`starts`, `lengths`, `counts`,
`indices`) describe the packing, so heads operate on `slot.m` rows and gather
their readout tokens through `indices`. `Slot.attend` applies RoPE while writing
the head-major layout, then runs one fused tensor-core attention kernel for every
length and window. It needs no scratch buffers; keys outside a question's span
or local window are masked, and padding tokens return zeros.
