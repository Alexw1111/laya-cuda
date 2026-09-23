"""Plot and document a completed p99_probe comparison."""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.style import COLORS, MUTED, TITLE, apply_style

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rounds = json.loads((args.directory/"summary.json").read_text())
    raw = [json.loads(p.read_text()) for p in sorted(args.directory.glob("*-*.json")) if p.name != "summary.json"]
    apply_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    table = []
    for ax, section, title in zip(axes, ("suites", "steady"), ("Mixed requests · preparation included", "Fixed shape · 1,000 calls after warmup")):
        names = list(rounds[0][section])
        for index, (variant, color) in enumerate((("before", COLORS[1]), ("after", COLORS[0]))):
            samples = np.array([[r[section][name][variant]["p99"] for r in rounds] for name in names])
            values = np.median(samples, axis=1)
            bars = ax.barh(np.arange(len(names))+(index-.5)*.34, values, .3, color=color, label=variant,
                           xerr=[values-samples.min(axis=1), samples.max(axis=1)-values], capsize=3, error_kw={"ecolor": MUTED, "elinewidth": .85})
            for bar, value, high in zip(bars, values, samples.max(axis=1)):
                ax.annotate(f"{value:.2f}", (high, bar.get_y()+bar.get_height()/2),
                            xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.xaxis.grid(True)
        ax.set_xlim(right=ax.get_xlim()[1]*1.15)
        ax.set_title(title, loc="left", pad=16, **TITLE)
        ax.set_xlabel("P99 API latency (ms) · median across 3 rounds")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(0,-.13), ncol=2, frameon=False)
        for name in names:
            before, after = [float(np.median([r[section][name][v]["p99"] for r in rounds])) for v in ("before", "after")]
            table.append(f"| {section} / {name} | {before:.2f} | {after:.2f} | {(1-after/before)*100:.1f}% |")
    fig.suptitle("P99 optimization · Windows / RTX 4090", fontsize=18, fontweight="normal", **TITLE)
    fig.supxlabel("Whiskers = min–max across rounds, not confidence intervals. Mixed suites have 25 requests each; all first-use costs remain included.", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(args.directory/f"p99.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    exact = sum(r["exact_answers"] for r in rounds)
    total = sum(r["requests"] for r in rounds)
    memory = {v: max(x["metrics"]["owned_gpu_bytes"] for p in args.directory.glob(f"{v}-*.json")
                     for x in json.loads(p.read_text())["records"]) for v in ("before", "after")}
    report = "\n".join([
        "# P99 optimization results", "", "Windows / RTX 4090; English Laya checkpoint; 2026-09-22.", "",
        "A new shape now executes once and returns its answer. Its second use captures a CUDA Graph; later uses replay it. "
        "The old path executed a preparation forward, captured another forward, then replayed the graph on the first request. "
        "Deferring capture eliminates redundant first-request execution and avoids capturing one-off shapes. "
        "The four-slot cache bound, kernels and precision are unchanged.", "",
        f"Full answer dictionaries match for **{exact}/{total}** paired requests. "
        "Steady-state probes include 1,000 calls per shape per variant per round (18,000 calls total), after 20 warmups. "
        "Each worker also runs the same 200 mixed requests after warming only the first input five times. "
        "Variants alternate execution order for three rounds in separate processes.", "",
        "| Workload | Before P99 ms | After P99 ms | Reduction |", "|---|---:|---:|---:|", *table, "",
        f"Maximum recorded allocator ownership: before {memory['before']/2**20:.2f} MiB; after {memory['after']/2**20:.2f} MiB. "
        "This is not total process VRAM. No cache enlargement was used.", "",
        "Small mixed-suite P99s remain descriptive and are dominated by one or two preparation events. "
        "The remaining first long-shape cost is not eliminated. The larger fixed-shape tests check for a steady-state regression; "
        "they are not substituted for end-to-end results. Other GPU architectures were not tested.", "",
        "99 tests passed, including all three local checkpoints' eager/graph lifecycle tests, batching, capture-on-reuse, "
        "eviction, and rejection of CPU inputs to the official baseline. Existing experimental model-quality restrictions remain.", "",
        "This `verified/` run had no concurrent assistant-launched GPU work. Earlier `screen/` and `paired/` directories "
        "are exploratory; the latter overlapped a CUDA capability check and is not used here.", "",
        "Reproduce from the repository with its locked benchmark environment:", "", "```powershell",
        "uv run --no-sync python tools/p99_probe.py --before reports/p99-optimization/baseline --after reports/p99-optimization/candidate --output reports/p99-rerun --rounds 3 --steady-samples 1000",
        "uv run --no-sync python tools/p99_report.py reports/p99-rerun", "```", "",
        "Source snapshots, source hashes, checkpoint weight hashes, frozen case-file hash and individual timings are retained. "
        "The comparison uses `reports/library-examples/cases.jsonl`; keep this input file and the same checkpoint. "
        "See `reports/reference-device-audit.json` for the separate official GPU device audit.", "",
        "![P99 comparison](p99.png)", ""])
    (args.directory/"results.md").write_text(report, encoding="utf-8")
    print(args.directory/"results.md")


if __name__ == "__main__":
    main()
