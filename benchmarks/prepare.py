"""Freeze upstream Laya benchmark inputs as backend-independent JSONL."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from .backends import UPSTREAM_REVISION

DATASETS = {"ag_news": ("fancyzhx/ag_news", None), "emotion": ("dair-ai/emotion", "split"),
            "banking77": ("mteb/banking77", None), "massive": ("mteb/amazon_massive_intent", None)}
SOURCE = f"https://github.com/nandhakishorm/laya/tree/{UPSTREAM_REVISION}/research/scripts"


def classification_case(suite, row, labels, rng=None):
    """Question text/order follows bench_apps.py or bench_local.py (Apache-2.0)."""
    if suite == "ag_news":
        criteria = dict(zip(["world", "sports", "business", "sci_tech"],
                            ["world news and international politics", "sports", "business and economy", "science and technology"]))
        state, qid, instruction, gold = {"article": row["text"]}, "topic", "What is the topic of `article`?", list(criteria)[row["label"]]
    elif suite == "emotion":
        criteria = {n: None for n in ["sadness", "joy", "love", "anger", "fear", "surprise"]}
        state, qid, instruction, gold = {"text": row["text"]}, "emotion", "Which emotion is most strongly expressed in `text`?", list(criteria)[row["label"]]
    elif suite == "banking77":
        criteria = {label.replace("_", " "): None for label in labels}
        state, qid, instruction, gold = {"message": row["text"]}, "intent", "Which banking intent does `message` express?", row["label_text"].replace("_", " ")
    else:
        keys = [row["label_text"]]+rng.sample([label for label in labels if label != row["label_text"]], min(19, len(labels)-1))
        rng.shuffle(keys)
        criteria = {key: key.replace("_", " ").replace(".", ": ") for key in keys}
        state, qid, instruction, gold = {"utterance": row["text"]}, "intent", "What is the user asking for in `utterance`?", row["label_text"]
    return {"state": state, "questions": {qid: {"type": "choice", "instructions": instruction, "criteria": criteria}},
            "gold": {qid: {"label": gold}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", choices=[*DATASETS, "typed_decisions"], default=["ag_news", "emotion", "banking77", "typed_decisions"])
    parser.add_argument("--per-suite", type=int, default=100)
    parser.add_argument("--languages", nargs="+", default=["en", "zh-CN"])
    parser.add_argument("--typed-data", type=Path, default=Path("benchmarks/data/typed-decisions.parquet"))
    parser.add_argument("--option-permutations", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.per_suite < 1 or args.output.exists():
        parser.error("Use a positive sample count and a new output path")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    rows, provenance = [], []
    for suite in args.suites:
        if suite == "typed_decisions":
            import pyarrow.parquet as pq
            dataset = pq.read_table(args.typed_data).to_pylist()[:args.per_suite]
            for i, row in enumerate(dataset):
                state = row["state"]
                try:
                    state = json.loads(state)
                except (ValueError, TypeError):
                    pass
                rows.append({"id": f"typed_decisions/{i}", "suite": suite, "state": state,
                             "questions": json.loads(row["questions"]), "gold": json.loads(row["gold"])})
            provenance.append({"suite": suite, "path": str(args.typed_data),
                               "sha256": hashlib.sha256(args.typed_data.read_bytes()).hexdigest(),
                               "labels": "Synthetic teacher labels; not human-verified ground truth"})
            continue
        repo, config = DATASETS[suite]
        revision = HfApi().dataset_info(repo).sha
        for language in args.languages if suite == "massive" else [config]:
            dataset = load_dataset(repo, language, split="test", revision=revision)
            labels = sorted(set(dataset["label_text"])) if suite in ("banking77", "massive") else []
            rng = random.Random(13)
            name = f"massive.{language}" if suite == "massive" else suite
            for i, row in enumerate(dataset.select(range(min(args.per_suite, len(dataset))))):
                rows.append({"id": f"{name}/{i}", "suite": name,
                             **classification_case(suite, row, labels, rng)})
            provenance.append({"suite": name, "dataset": repo, "revision": revision, "config": language,
                               "split": "test", "selection": "first N, matching upstream", "seed": 13,
                               "fingerprint": dataset._fingerprint, "in_training_per_upstream": suite == "ag_news"})
    if args.option_permutations:
        variants = []
        for row in rows:
            variant = json.loads(json.dumps(row))
            changed = False
            for question in variant["questions"].values():
                if question["type"] == "choice" and isinstance(question["criteria"], dict):
                    question["criteria"] = dict(reversed(list(question["criteria"].items())))
                    changed = True
            if changed:
                variant["id"] += "/reversed"
                variant["suite"] += ".reversed"
                variants.append(variant)
        rows.extend(variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    manifest = {"upstream": SOURCE, "upstream_revision": UPSTREAM_REVISION, "datasets": provenance,
                "cases": len(rows), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "note": "Adapted input builders, not upstream raw-logit execution or recalibration. All backends use public prediction APIs."}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
