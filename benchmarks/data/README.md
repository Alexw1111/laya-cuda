# Fixed quality fixture

Source license: Apache-2.0, declared in the pinned dataset card. Attribution: LocalLLaMA, Typed Decisions dataset.

`typed-decisions.parquet` contains the 400-row test split from `LocalLLaMA/typed-decisions`, configuration `all`, revision `ea9306458d6e9563628369a3d1e72e362fb381d2`:

https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/ea9306458d6e9563628369a3d1e72e362fb381d2/all/test-00000-of-00001.parquet

SHA-256: `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`.

It contains 2,000 typed decisions. Labels are synthetic/pseudo-labels, not independently human-verified ground truth. The typed-decisions model is associated with this dataset; this test split does not establish general-domain equivalence. English and multilingual models use the same fixture for matched comparisons, not comprehensive language evaluation.

The fixture was frozen before quality evaluation. Development probes used repeated payment/refund text and primitive random tensors, not these answers. Keep tuning and final evaluation separate. Do not loosen numerical thresholds after observing results. One exception is recorded: on 2026-09-24 the accuracy criterion became a sign test after a one-decision deficit was observed; see [the FP32-reference report](../../reports/fp32-reference/results.md). Preserve dataset provenance and license when redistributing this fixture.
