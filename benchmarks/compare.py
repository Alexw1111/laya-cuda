"""Process-isolated task comparison of laya-cuda, pinned upstream Laya and Jev."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from .backends import JEV_MODEL, open_backend, validate_answers


def task_metrics(records):
    correct, total, confidence, errors, brier, nll, score_errors = 0, 0, [], [], [], [], []
    unavailable_predictions = 0
    for record in records:
        for key, gold in record["gold"].items():
            total += 1
            if "error" in record:
                continue
            answer = record["answers"][key]
            if answer["type"] == "noul":
                probabilities = {"false": 1-answer["noul"], "true": answer["noul"]}
                prediction = "true" if answer["noul"] >= .5 else "false"
            else:
                probabilities = answer.get("probabilities")
                prediction = answer.get("choice")
                if prediction is None and probabilities:
                    prediction = max(probabilities, key=probabilities.get)
                if answer["type"] == "score":
                    target = gold.get("score", gold.get("label"))
                    score_errors.append(abs(answer["score"]-float(target)))
                    if prediction is None:
                        unavailable_predictions += 1
            target = str(gold["label"])
            hit = prediction == target
            correct += hit
            if probabilities is not None:
                confidence.append(max(probabilities.values()))
                errors.append(int(hit))
                brier.append(sum((p-int(k == target))**2 for k, p in probabilities.items()))
                nll.append(-math.log(max(probabilities.get(target, 0), 1e-12)))
    ece = None
    if confidence:
        c, a = np.array(confidence), np.array(errors)
        bins = np.minimum((c*15).astype(int), 14)
        ece = sum(float((bins == i).mean())*abs(float(c[bins == i].mean()-a[bins == i].mean())) for i in range(15) if (bins == i).any())
    return {"decisions_including_failures": total,
            "accuracy_including_failures": correct/total if total and not unavailable_predictions else None,
            "unavailable_class_predictions": unavailable_predictions,
            "probabilistic_decisions": len(confidence), "brier_sum_per_decision": float(np.mean(brier)) if brier else None,
            "nll": float(np.mean(nll)) if nll else None, "ece_15_bins": ece,
            "score_mae": float(np.mean(score_errors)) if score_errors else None,
            "failed_requests": sum("error" in r for r in records)}


def worker(args):
    import psutil
    cases = [json.loads(line) for line in args.cases_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        cases = cases[:args.limit]
    if not cases or len({r["id"] for r in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs")
    started = time.perf_counter()
    with open_backend(args.worker, args.checkpoint, upstream=args.upstream, model=args.jev_model,
                      max_requests=len(cases)+args.warmup, max_cost=args.max_cost) as model:
        record = {"backend": args.worker, "checkpoint": args.checkpoint, "round": args.round,
                  "cases_sha256": hashlib.sha256(args.cases_file.read_bytes()).hexdigest(),
                  "platform": platform.platform(), "load_ms": (time.perf_counter()-started)*1000,
                  "upstream": getattr(model, "comparison_provenance", None),
                  "warmup": args.warmup, "limit": args.limit, "records": [], "warmup_usage": [], "warmup_ms": [],
                  "python": platform.python_version(),
                  "runtime_device": getattr(getattr(model, "runtime", None), "device_info", None),
                  "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                               if d.metadata["Name"] in ("laya-cuda", "numpy", "cupy-cuda12x", "cupy-cuda13x")}}
        import laya_cuda
        record["implementation"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in Path(laya_cuda.__file__).parent.iterdir() if p.suffix in (".py", ".cu", ".json")}
        try:
            record["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            record["gpu"] = None
        if hasattr(model, "path"):
            from .reproduce import checkpoint_files
            record["checkpoint_files"] = checkpoint_files(model.path)
            record["checkpoint_sha256"] = record["checkpoint_files"]["model.safetensors"]
        for _ in range(args.warmup):
            begin = time.perf_counter()
            response = model.predict(cases[0]["state"], cases[0]["questions"])
            record["warmup_ms"].append((time.perf_counter()-begin)*1000)
            record["warmup_usage"].append(response.get("usage"))
        for case in cases:
            row = {"id": case["id"], "suite": case["suite"], "gold": case["gold"],
                   "input_sha256": hashlib.sha256(json.dumps([case["state"], case["questions"]], ensure_ascii=False).encode()).hexdigest()}
            begin = time.perf_counter()
            try:
                response = validate_answers(model.predict(case["state"], case["questions"]), case["questions"])
                row.update(answers=response["answers"], usage=response.get("usage"), model=response.get("model"),
                           metrics=model.last_metrics.copy())
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["ms"] = (time.perf_counter()-begin)*1000
            record["records"].append(row)
            record["rss_bytes"] = max(record.get("rss_bytes", 0), psutil.Process().memory_info().rss)
            args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.worker == "cuda" and {"torch", "transformers"} & sys.modules.keys():
            raise RuntimeError("Lightweight worker imported heavyweight dependencies")


def summarize(directory):
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*-r*.json"))]
    if len({r["cases_sha256"] for r in runs}) != 1:
        raise ValueError("Comparison input files differ")
    aligned = [[(x["id"], x["input_sha256"]) for x in r["records"]] for r in runs]
    if any(ids != aligned[0] for ids in aligned[1:]):
        raise ValueError("Comparison requests differ")
    if not aligned[0]:
        raise ValueError("No measured requests")
    status_path = directory/"experiment.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        actual = [(r["backend"], r["round"]) for r in runs]
        expected = [(b, n) for b in status["backends"] for n in range(status["rounds"])]
        if sorted(actual) != sorted(expected) or any(len(r["records"]) != status["requests_per_worker"] for r in runs):
            raise ValueError("Incomplete or duplicate worker results; report refused")
        from .reproduce import digest
        for name, expected_hash in status.get("result_hashes", {}).items():
            if digest(directory/name) != expected_hash:
                raise ValueError(f"Recorded result changed: {name}")
    result = {"note": "Local API and remote network latency are separate deployment costs. No verified-speedup promotion.", "backends": {}}
    for backend in sorted({r["backend"] for r in runs}):
        selected = sorted((r for r in runs if r["backend"] == backend), key=lambda r: r.get("round", 0))
        suites = {}
        for suite in sorted({row["suite"] for row in selected[0]["records"]}):
            rounds = []
            for run in selected:
                rows = [row for row in run["records"] if row["suite"] == suite]
                times = [row["ms"] for row in rows if "error" not in row]
                rounds.append({"round": run.get("round", 0), **task_metrics(rows), "p50_ms": float(np.median(times)) if times else None,
                               "p95_ms": float(np.percentile(times, 95)) if times else None,
                               "p99_ms": float(np.percentile(times, 99)) if times else None,
                               "requests_per_second_including_failures": len(rows)/(sum(r["ms"] for r in rows)/1000),
                               "request_latency_ms_including_failures": [r["ms"] for r in rows]})
            suites[suite] = rounds
        result["backends"][backend] = suites
    (directory/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=["cuda", "official", "upstream", "jev"], default=["cuda", "upstream"])
    parser.add_argument("--checkpoint", default="models/laya")
    parser.add_argument("--checkpoint-source", help="Pinned registry alias used to obtain matching weights on another machine")
    parser.add_argument("--upstream", default="vendor/laya-upstream")
    parser.add_argument("--jev-model", default=JEV_MODEL)
    parser.add_argument("--max-cost", type=float, default=1.0, help="Reported-cost stop per Jev worker, USD; not a hard billing cap")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--worker", choices=["cuda", "official", "upstream", "jev"])
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 0 or args.limit is not None and args.limit < 1:
        parser.error("Invalid run counts")
    if len(set(args.backends)) != len(args.backends):
        parser.error("Backend names must be unique")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.iterdir()):
        parser.error("Use a fresh output directory")
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        parser.error('Install the benchmark extra to generate reports')
    from .reproduce import digest, freeze, verify
    bundle = freeze(args)
    args.cases_file = (bundle/"inputs.jsonl").resolve()
    if Path(args.checkpoint).is_dir():
        args.checkpoint = str(Path(args.checkpoint).resolve())
    args.upstream = str(Path(args.upstream).resolve())
    case_count = len([line for line in args.cases_file.read_text(encoding="utf-8").splitlines() if line.strip()])
    status = {"status": "running", "backends": args.backends, "rounds": args.rounds,
              "requests_per_worker": min(args.limit, case_count) if args.limit else case_count}
    status_path = args.output/"experiment.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    for r in range(args.rounds):
        for backend in args.backends if r % 2 == 0 else args.backends[::-1]:
            command = [sys.executable, "-m", "benchmarks.compare", "--worker", backend, "--round", str(r),
                       "--cases-file", str(args.cases_file.resolve()), "--checkpoint", args.checkpoint,
                       "--upstream", args.upstream, "--jev-model", args.jev_model, "--max-cost", str(args.max_cost),
                       "--warmup", str(args.warmup), "--output", str((args.output/f"{backend}-r{r}.json").resolve())]
            if args.limit:
                command += ["--limit", str(args.limit)]
            try:
                subprocess.run(command, check=True, cwd=bundle/"project")
            except subprocess.CalledProcessError:
                status["status"] = f"incomplete · {backend} round {r} failed"
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                raise
    manifest = verify(bundle)
    for path in args.output.glob("*-r*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["backend"] != "jev" and run["checkpoint_files"] != manifest["checkpoint"]["files"]:
            raise ValueError("Checkpoint changed during the experiment")
    summarize(args.output)
    status["status"] = "complete · input and checkpoint bytes verified"
    status["result_hashes"] = {p.name: digest(p) for p in args.output.glob("*-r*.json")}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    from .report import render
    print(render(args.output))


if __name__ == "__main__":
    main()
