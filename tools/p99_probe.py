"""Balanced before/after API tail-latency comparison, retaining cold-shape costs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    # Otherwise the import below silently falls back to the installed package.
    if not (args.source/"laya_cuda").is_dir():
        raise SystemExit(f"{args.source} has no laya_cuda package (a snapshot from before the rename?)")
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    from laya_cuda import Engine
    cases = [json.loads(line) for line in args.cases_file.read_text(encoding="utf-8").splitlines()]
    result = {"source": str(args.source.resolve()), "implementation": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.source/"laya_cuda").glob("*") if p.is_file()},
        "input_sha256": hashlib.sha256(args.cases_file.read_bytes()).hexdigest(),
        "steady_samples": args.steady_samples, "records": [], "steady": {}}
    start = time.perf_counter()
    with Engine(args.checkpoint) as model:
        result["load_ms"] = (time.perf_counter()-start)*1000
        import cupy as cp
        result["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        result["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                              if d.metadata["Name"] in ("numpy", "cupy-cuda12x", "cupy-cuda13x")}
        with (model.path/"model.safetensors").open("rb") as stream:
            result["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        for _ in range(5):
            model.predict(cases[0]["state"], cases[0]["questions"])
        for case in cases:
            start = time.perf_counter()
            response = model.predict(case["state"], case["questions"])
            elapsed = (time.perf_counter()-start)*1000
            result["records"].append({"id": case["id"], "suite": case["suite"], "ms": elapsed,
                                      "answers": response["answers"], "metrics": model.last_metrics.copy()})
        selected = [cases[0], cases[1], next(c for c in cases if c["suite"] == "typed_decisions")]
        for case in selected:
            for _ in range(20):
                model.predict(case["state"], case["questions"])
            times = []
            for _ in range(args.steady_samples):
                start = time.perf_counter()
                model.predict(case["state"], case["questions"])
                times.append((time.perf_counter()-start)*1000)
            result["steady"][case["id"]] = {"ms": times, "p50": float(np.median(times)),
                "p99": float(np.percentile(times, 99)), "shape": model.last_metrics["shape"]}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path, default=Path.cwd())
    parser.add_argument("--before-python", default=sys.executable)
    parser.add_argument("--after-python", default=sys.executable)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", default="models/laya")
    parser.add_argument("--cases-file", type=Path, default=Path("reports/library-examples/cases.jsonl"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steady-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.source:
        return worker(args)
    if not args.before:
        parser.error("--before source snapshot is required")
    args.output.mkdir(parents=True, exist_ok=False)
    for r in range(args.rounds):
        for name in ("before", "after") if r % 2 == 0 else ("after", "before"):
            subprocess.run([getattr(args, name+"_python"), str(Path(__file__).resolve()), "--source", str(getattr(args, name).resolve()),
                            "--checkpoint", args.checkpoint, "--cases-file", str(args.cases_file.resolve()),
                            "--steady-samples", str(args.steady_samples), "--output", str(args.output/f"{name}-{r}.json")], check=True)
    import numpy as np
    summary = []
    for r in range(args.rounds):
        before, after = [json.loads((args.output/f"{name}-{r}.json").read_text()) for name in ("before", "after")]
        assert before["input_sha256"] == after["input_sha256"]
        assert [x["id"] for x in before["records"]] == [x["id"] for x in after["records"]]
        exact = sum(a["answers"] == b["answers"] for a,b in zip(before["records"],after["records"]))
        entry = {"round": r, "exact_answers": exact, "requests": len(before["records"]), "suites": {}, "steady": {}}
        for suite in sorted({v["suite"] for v in before["records"]}):
            entry["suites"][suite] = {name: {"p50": float(np.median([x["ms"] for x in data["records"] if x["suite"] == suite])),
                                                  "p99": float(np.percentile([x["ms"] for x in data["records"] if x["suite"] == suite], 99))}
                                      for name, data in (("before",before),("after",after))}
        entry["steady"] = {case: {name: {k: v for k,v in data["steady"][case].items() if k != "ms"}
                                  for name,data in (("before",before),("after",after))} for case in before["steady"]}
        summary.append(entry)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
