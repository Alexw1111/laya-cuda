"""Render completed benchmark evidence without inventing missing results."""
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.run import TIMED, summarize

lines=["# Platform and workload comparison", "",
       "Generated from raw, synchronized API measurements. All times are milliseconds. Three rounds per backend; each scenario uses 20 warmups and 200 observations. P50 columns show the median of the three per-round P50 values. Tail ratios are the worst round's CUDA / official BF16 ratio; lower is better.", "",
       "A verified speedup also requires all three rounds to pass the latency gates, scenario numerical checks and the model's task-quality gate. Pseudo-label accuracy is not comprehensive model quality. See [validation](validation.md) for scope, installation limitations and references.", ""]
for platform in ("windows","wsl"):
    directory=Path("reports")/f"{platform}-benchmark"
    if not directory.exists():
        continue
    summary=summarize(directory)
    runs=[json.loads(p.read_text()) for p in directory.glob("*-r*.json")]
    lines += [f"## {platform}",""]
    for model in sorted({r["model"] for r in runs}):
        groups={mode:sorted([r for r in runs if r["model"]==model and r["mode"]==mode],key=lambda r:r["round"]) for mode in TIMED}
        if any(len(v)!=3 for v in groups.values()) or model not in summary["models"]:
            lines += [f"{model}: incomplete; no three-round claim.",""]
            continue
        record=summary["models"][model]
        quality=record["quality"]
        lines += [f"### {model}","",
            f"Against SDK FP32: probability MAE {quality['mae']:.6f}, maximum {quality['max']:.6f}; decision agreement {quality['agreement']:.2%}. Accuracy: CUDA {quality['accuracy']['cuda']:.2%}, SDK FP32 {quality['accuracy']['official-fp32']:.2%}, SDK FP16 {quality['accuracy']['official-fp16']:.2%}, SDK BF16 {quality['accuracy']['official']:.2%}. Quality gate: **{'pass' if quality['passed'] else 'fail'}**.","",
            "| Workload | CUDA P50 | SDK BF16 P50 | SDK FP16 P50 | Worst P95 ratio | Worst P99 ratio | Verified speedup |",
            "|---|---:|---:|---:|---:|---:|---|"]
        for case,stats in record["timing"].items():
            p50={mode:statistics.median(r["timing"][case]["p50"] for r in group) for mode,group in groups.items()}
            lines.append(f"| {case} | {p50['cuda']:.3f} | {p50['official']:.3f} | {p50['official-fp16']:.3f} | {max(r['p95_ratio'] for r in stats['rounds']):.3f} | {max(r['p99_ratio'] for r in stats['rounds']):.3f} | {'yes' if stats['verified_speedup'] else 'no'} |")
        lines += ["", "| Resource / startup | CUDA | SDK BF16 | SDK FP16 |", "|---|---:|---:|---:|"]
        for name,fn in (
            ("Median model load ms",lambda rs: statistics.median(r["load_ms"] for r in rs)),
            ("First 32-token / 1-question API ms (round 0)",lambda rs: rs[0]["timing"]["l32-q1"]["first_ms"]),
            ("Maximum sampled RSS MiB",lambda rs: max(v["rss_bytes"] for r in rs for v in r["timing"].values())/2**20),
        ):
            values=[fn(groups[m]) for m in ("cuda","official","official-fp16")]
            lines.append(f"| {name} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |")
        owned=max(v["metrics"]["owned_gpu_bytes"] for r in groups["cuda"] for v in r["timing"].values())/2**20
        lines += ["",f"Maximum sampled CuPy-owned GPU pool: {owned:.1f} MiB. This is allocator ownership, not process total or a peak measurement. Complete per-round P50/P95/P99, individual observations, throughput, device timings and cache-miss metrics are in [{platform} raw run files]({platform}-benchmark/); [promotion decisions]({platform}-benchmark/summary.json) are separate.",""]
lines += ["## All-model correctness preflight", "",
          "These repeated-token length probes use one observation, not a latency benchmark. They are stress diagnostics: errors are against SDK FP32, shown beside the SDK's own FP16 error on the same input. The long-input suite and the fixed set are the gates. Raw records also contain all returned answers.", ""]
for platform in ("windows","wsl"):
    directory=Path("reports")/f"{platform}-current"
    if not directory.exists():
        continue
    summary=summarize(directory)
    for model,record in summary["models"].items():
        q=record["quality"]
        lines += [f"### {platform}: {model}", "",
                  f"Dataset vs SDK FP32: MAE {q['mae']:.6f}, max {q['max']:.6f}, agreement {q['agreement']:.2%}; CUDA accuracy {q['accuracy']['cuda']:.2%}, SDK FP32 {q['accuracy']['official-fp32']:.2%}. Dataset gate: {'pass' if q['passed'] else 'fail'}."]
        if record["long"]:
            g=record["long"]
            lines.append(f"Long inputs ({g['decisions']} decisions, {g['tokens'][0]}-{g['tokens'][1]} tokens) vs SDK FP32: MAE {g['mae']:.6f}, max {g['max']:.6f}, agreement {g['agreement']:.2%}; SDK FP16 max {g['peers']['official-fp16']['max']:.6f}, SDK BF16 max {g['peers']['official']['max']:.6f}. Long-input gate: {'pass' if g['passed'] else 'fail'}.")
        lines += ["", "| Stress probe | CUDA MAE | CUDA max | SDK FP16 max | Labels match FP32 |",
                  "|---|---:|---:|---:|---|"]
        for case,s in record["timing"].items():
            lines.append(f"| {case} | {s['probability_mae']:.6f} | {s['probability_max']:.6f} | {s['official_fp16_probability_max']:.6f} | {'yes' if s['labels_match_reference'] else 'no'} |")
        lines.append("")
Path("reports/comparison.md").write_text("\n".join(lines),encoding="utf-8")
print("reports/comparison.md")
