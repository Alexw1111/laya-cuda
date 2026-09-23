# Upstream benchmark attribution

The input builders in `prepare.py` adapt selected question definitions and
sample-selection rules from NandhaKishorM/laya, Apache License 2.0:

- Repository: https://github.com/nandhakishorm/laya
- Revision: `573e5b62696ba441230cd6be71d593331b5d23af`
- Authors: Convai Innovations and the repository contributors.
- Files: `research/scripts/bench_apps.py`, `research/scripts/bench_local.py`.
- License: https://github.com/nandhakishorm/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/LICENSE

Changes: standalone JSONL inputs, pinned dataset revisions, explicit sample
limits, optional reversed-choice variants, and local typed-decisions parquet.
We compare public prediction APIs rather than copying upstream raw-logit batching
or fitting calibration temperatures. AG News, Emotion, Banking77, MASSIVE and
typed-decisions are implemented; the other upstream suites are not yet ported.
Upstream published Jev/T4/CPU scores are not treated as measurements from this
project. Dataset licenses and training overlap remain those of each dataset.
