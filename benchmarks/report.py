"""Render portable task-comparison charts and a Markdown report."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from .style import BACKEND_COLORS, COLORS, GRID, MUTED, TITLE, apply_style


NAMES = {"cuda": "laya-cuda", "official": "official SDK", "upstream": "upstream SDK", "jev": "Jev · OpenRouter API"}


def overview(plt, groups, latency=None, suite=None):
    """P50/P95 request latency and accuracy for each (model, backend); a remote backend appears once.

    `latency`, aligned with `groups`, supplies local latency from other runs (for example one request
    repeated), restricted to `suite` when given. Accuracy always comes from `groups`.
    """
    from matplotlib.ticker import PercentFormatter
    from .compare import task_metrics
    rows, remote, headline = [], {}, []
    for index, (model, runs) in enumerate(groups):
        speed = {}
        for backend in NAMES:
            selected = [r for r in runs if r["backend"] == backend]
            if not selected or backend in remote:
                continue
            local = latency is not None and backend != "jev"
            timed = [r for r in latency[index] if r["backend"] == backend] if local else selected
            times = [x["ms"] for r in timed for x in r["records"]
                     if "error" not in x and (not local or suite is None or x["suite"] == suite)]
            speed[backend] = (float(np.median(times)), float(np.percentile(times, 95))) if times else (float("nan"),) * 2
            accuracy = float(np.mean([task_metrics(r["records"])["accuracy_including_failures"] or 0 for r in selected]))
            if backend == "jev":
                # Remote latency includes the network round trip; say so on the bar itself.
                name = NAMES[backend] + (" (mixed requests, incl. network)" if latency is not None else " (incl. network)")
            else:
                name = f"{model} · {NAMES[backend]}"
            row = (backend, name, speed[backend][0], accuracy, speed[backend][1])
            if backend == "jev":
                remote[backend] = row
            else:
                rows.append(row)
        official, cuda = speed.get("official", (np.nan,) * 2), speed.get("cuda", (np.nan,) * 2)
        if np.isfinite(official[0] / cuda[0]):
            headline.append(f"{model} {official[0]/cuda[0]:.1f}× ({official[1]/cuda[1]:.1f}×)")
    rows += remote.values()
    finite = [v for r in rows for v in (r[2], r[4]) if np.isfinite(v)] or [1.0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 1.4+.48*len(rows)), layout="constrained")
    colors = [BACKEND_COLORS.get(r[0], COLORS[3]) for r in rows]
    # P95 as a light bar behind the solid P50 bar.
    left.barh(range(len(rows)), [r[4] for r in rows], .62, color=colors, alpha=.3)
    for ax, column, xlabel, fmt in ((left, 2, "API latency per request (ms, log scale): solid P50, light P95 · lower is better", "{:.1f} ms"),
                                    (right, 3, f"Accuracy on the {'' if latency is not None else 'same '}frozen requests · higher is better", "{:.1%}")):
        ax.barh(range(len(rows)), [r[column] for r in rows], .62, color=colors)
        for i, r in enumerate(rows):
            known = np.isfinite(r[column])
            label = fmt.format(r[column]) if known else "N/A"
            if column == 2 and np.isfinite(r[4]):
                label = f"P50 {r[2]:.1f} ms · P95 {r[4]:.1f} ms"
            ax.annotate(label, (max(r[column], r[4]) if column == 2 and known else r[column] if known else min(finite)/2, i),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_ylim(len(rows)-.5, -.5)
        ax.xaxis.grid(True, color=GRID)
        ax.set_xlabel(xlabel)
    left.set_yticks(range(len(rows)), [r[1] for r in rows])
    right.set_yticks(range(len(rows)), [""]*len(rows))
    left.set_xscale("log")
    left.set_xlim(min(finite)/2, max(finite)*8)
    right.set_xlim(0, 1.12)
    right.xaxis.set_major_formatter(PercentFormatter(1))
    kind = "Same request repeated" if latency is not None else "Latency"
    title = f"{kind}, official SDK ÷ laya-cuda, P50 (P95): " + ", ".join(headline) if headline else "Latency and accuracy"
    fig.suptitle(title, fontsize=13, fontweight="normal", **TITLE)
    return fig


def observations(summary):
    return [{"backend": backend, "suite": suite, "round": index, **values}
            for backend, suites in summary["backends"].items()
            for suite, rounds in suites.items() for index, values in enumerate(rounds)]


def cache_diagnostics(runs):
    result = []
    for run in runs:
        for suite in sorted({r["suite"] for r in run["records"]}):
            rows = [r for r in run["records"] if r["suite"] == suite and "error" not in r]
            hit = [r["ms"] for r in rows if r.get("metrics", {}).get("cache_miss") is False]
            miss = [r["ms"] for r in rows if r.get("metrics", {}).get("cache_miss") is True]
            capture = [r["ms"] for r in rows if r.get("metrics", {}).get("graph_capture") is True]
            capture_known = any("graph_capture" in r.get("metrics", {}) for r in rows)
            slowest = max(rows, key=lambda r: r["ms"], default={})
            result.append({"backend": run["backend"], "suite": suite, "round": run.get("round", 0),
                           "hits": len(hit), "misses": len(miss), "unknown_cache": len(rows)-len(hit)-len(miss),
                           "graph_captures": len(capture) if capture_known else None,
                           "hit_p99_ms": float(np.percentile(hit, 99)) if hit else None,
                           "miss_p99_ms": float(np.percentile(miss, 99)) if miss else None,
                           "slowest_api_ms": slowest.get("ms"),
                           "gpu_at_slowest_ms": slowest.get("metrics", {}).get("gpu_ms")})
    return result


def render(directory, note=""):
    from .compare import summarize
    summary = summarize(directory)
    rows = observations(summary)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*-r*.json"))]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    apply_style(plt)
    backends = list(summary["backends"])
    suites = sorted({r["suite"] for r in rows})
    plots = []

    def save(fig, name):
        gpu = next((r["gpu"] for r in runs if r.get("gpu")), "GPU metadata unavailable")
        fig.supxlabel(f"{gpu}  |  {len(runs[0]['records'])} requests / worker  |  {len(runs)} worker runs\n"
                      "Cache misses included · successful-request latency · descriptive tails, not speedup approval",
                      fontsize=8, color=MUTED)
        fig.savefig(directory/f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(directory/f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        plots.append(name)

    def grouped(ax, categories, values, xlabel, percent=False):
        height = .72/len(backends)
        maximum = 0
        for i, backend in enumerate(backends):
            medians, lower, upper = [], [], []
            for category in categories:
                samples = [v for v in values(backend, category) if v is not None and np.isfinite(v)]
                median = float(np.median(samples)) if samples else float("nan")
                medians.append(median)
                lower.append(median-min(samples) if samples else 0)
                upper.append(max(samples)-median if samples else 0)
                maximum = max(maximum, max(samples) if samples else 0)
            positions = np.arange(len(categories))+(i-(len(backends)-1)/2)*height
            ax.barh(positions, medians, height*.84, label=backend, color=BACKEND_COLORS.get(backend, COLORS[i % len(COLORS)]),
                           xerr=np.array([lower, upper]), capsize=2, error_kw={"elinewidth": .85, "ecolor": MUTED})
            for pos, val, high in zip(positions, medians, upper):
                label = "N/A" if not np.isfinite(val) else f"{val:.0%}" if percent else f"{val:.2f}"
                ax.annotate(label, (val+high if np.isfinite(val) else 0, pos), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(range(len(categories)), [s.replace(".reversed", " · reversed") for s in categories])
        ax.set_ylim(len(categories)-.5, -.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, 1.15 if percent else max(maximum*1.24, 1))
        if percent:
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper left", bbox_to_anchor=(0, -.17), ncol=4, frameon=False, fontsize=8)

    fig = overview(plt, [(Path(runs[0].get("checkpoint", "model")).name, runs)])
    save(fig, "overview")

    fig, axes = plt.subplots(1, 3, figsize=(17, max(5, len(suites)*.82)), layout="constrained")
    for ax, metric, title in zip(axes, ("p50_ms", "p95_ms", "p99_ms"), ("Median · P50", "Tail · P95", "Tail · P99")):
        grouped(ax, suites, lambda b, s: [r[metric] for r in rows if r["backend"] == b and r["suite"] == s], "Prediction latency (ms)")
        ax.set_title(title, loc="left", pad=18, **TITLE)
    fig.suptitle("Prediction API latency  |  median across rounds; whiskers = min–max", fontsize=17, fontweight="normal", **TITLE)
    save(fig, "latency")

    fig, ax = plt.subplots(figsize=(12, max(4, len(suites)*.72)), layout="constrained")
    grouped(ax, suites, lambda b, s: [r["accuracy_including_failures"] for r in rows if r["backend"] == b and r["suite"] == s],
            "Accuracy · failed requests count as incorrect", percent=True)
    ax.set_title("Decision quality  |  fixed evaluation inputs", loc="left", pad=20, **TITLE)
    save(fig, "quality")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), layout="constrained")
    for ax, field, scale, title, unit in zip(axes, ("load_ms", "rss_bytes", "owned_gpu_bytes"),
                                           (1000, 2**20, 2**20), ("Model loading", "Sampled process RAM", "Owned GPU allocations"),
                                           ("Seconds · excludes warmup", "MiB · sampled, not peak", "MiB · CUDA only; not total VRAM")):
        def values(backend, _, field=field, scale=scale):
            result = []
            for run in runs:
                if run["backend"] != backend:
                    continue
                value = run.get(field)
                if field == "owned_gpu_bytes":
                    value = max((r.get("metrics", {}).get(field, 0) for r in run["records"]), default=0) or None
                result.append(value/scale if value is not None else None)
            return result
        grouped(ax, ["All suites"], values, unit)
        ax.set_title(title, loc="left", pad=18, **TITLE)
    save(fig, "resources")

    fields = [k for k in rows[0] if k != "request_latency_ms_including_failures"]
    with (directory/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    diagnostics = cache_diagnostics(runs)
    with (directory/"cache-diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    write_report(directory, note)
    if (directory/"reproduce/manifest.json").exists():
        from .reproduce import pack
        pack(directory)
    return directory/"report.md"


CAPTIONS = {"overview": "Speed and accuracy", "latency": "Prediction latency", "quality": "Decision quality",
            "resources": "Resource footprint"}


def write_report(directory, note=""):
    """Write report.md from recorded runs and the charts already in the directory; never re-plots."""
    from .compare import summarize
    summary = summarize(directory)
    rows = observations(summary)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*-r*.json"))]
    diagnostics = cache_diagnostics(runs)
    status_path = directory/"experiment.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "legacy result: provenance incomplete"}
    if not note and (directory/"report-note.txt").exists():
        note = (directory/"report-note.txt").read_text(encoding="utf-8")
    if note:
        (directory/"report-note.txt").write_text(note, encoding="utf-8")

    def cell(value):
        # Escape markup so recorded text is shown literally in any Markdown viewer.
        return html.escape(str(value)).replace("|", "\\|")

    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"

    def table(header, values):
        return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                         + ["| " + " | ".join(cell(v) for v in row) + " |" for row in values])

    charts = "\n\n".join(f"### {CAPTIONS[name]}\n\n![{CAPTIONS[name]}]({name}.png)\n\n[SVG]({name}.svg) · [PNG]({name}.png)"
                         for name in CAPTIONS if (directory/f"{name}.png").exists())
    quality = table(["Backend", "Suite", "Round", "Decisions", "Failed requests", "Accuracy", "Probabilistic decisions", "Brier", "Score MAE"],
                    [(r["backend"], r["suite"], r["round"]+1, r["decisions_including_failures"], r["failed_requests"],
                      fmt(r["accuracy_including_failures"]), r["probabilistic_decisions"], fmt(r["brier_sum_per_decision"]),
                      fmt(r["score_mae"])) for r in rows])
    cache = table(["Backend", "Suite", "Round", "Hits", "Misses", "Unknown", "Graph captures", "Hit P99 ms", "Miss P99 ms",
                   "Slowest API ms", "GPU at slowest ms"],
                  [(r["backend"], r["suite"], r["round"]+1, r["hits"], r["misses"], r["unknown_cache"],
                    r["graph_captures"] if r["graph_captures"] is not None else "N/A", fmt(r["hit_p99_ms"]),
                    fmt(r["miss_p99_ms"]), fmt(r["slowest_api_ms"]), fmt(r["gpu_at_slowest_ms"])) for r in diagnostics])
    notice = f"> **{cell(status['status'])}**" + (f"\n>\n> {cell(note)}" if note else "")
    content = f"""# Laya benchmark report

Latency, decision quality and resource use on fixed evaluation inputs. Every chart traces back to recorded runs.

| Compared backends | Evaluation suites | Worker runs | Measured requests |
|---:|---:|---:|---:|
| {len(summary["backends"])} | {len({r["suite"] for r in rows})} | {len(runs)} | {sum(len(r["records"]) for r in runs):,} |

Data: [summary.json](summary.json) · [metrics.csv](metrics.csv) · [cache-diagnostics.csv](cache-diagnostics.csv) · [reproduction guide](reproduce/REPRODUCE.md)

{notice}

- Whiskers show variation across rounds, not confidence intervals. Latency quantiles cover successful requests only; failures remain visible below. P95/P99 are descriptive with small samples. Shape/cache misses are included.
- Model load time excludes process startup, warmup and uncached downloads. Local and remote latency have different costs. No verified speedup or numerical-equivalence claim is inferred from these charts.
- Repeated rounds reuse the same examples; they do not increase the independent quality sample size. AG News may overlap training; typed-decision labels are synthetic. Missing values are N/A.

## Charts

{charts}

## Quality and failure coverage

{quality}

## Why is P99 high?

Only the first input is repeated during warmup. New CUDA shapes allocate workspaces; graph construction also adds preparation cost. The runtime can defer capture until reuse, so a cache hit may still capture a graph. The four-slot LRU can evict earlier shapes. Inspect the capture counts when available; legacy runs do not record them. API latency includes preparation; device-event timing does not include all preparation. Their difference is not a CPU-only measurement.

With 25 requests, NumPy's interpolated P99 is 76% of the maximum plus 24% of the second-largest value. Use the full API result for end-to-end behavior. These hit/miss subsets diagnose it; they do not replace it with a better-looking number. Each subset has fewer samples and is not a steady-state guarantee. Unknown cache status is never treated as a hit.

{cache}

## Environment, protocol and fingerprints

Recorded in [experiment.json](experiment.json), [reproduce/manifest.json](reproduce/manifest.json) and each worker run: {" · ".join(f"[{r.name}]({r.name})" for r in sorted(directory.glob("*-r*.json")))}.
"""
    (directory/"report.md").write_text(content, encoding="utf-8")
    return directory/"report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="+", help="one result directory, or several with --overview")
    parser.add_argument("--note", default="")
    parser.add_argument("--overview", type=Path, metavar="STEM", help="write only a combined overview chart to STEM.png/.svg")
    parser.add_argument("--latency", type=Path, nargs="+", help="with --overview: local latency from these directories, one per directory")
    parser.add_argument("--suite", help="with --latency: only this suite's requests")
    args = parser.parse_args()
    if args.latency and len(args.latency) != len(args.directory):
        parser.error("--latency needs one directory for each result directory")
    if args.overview is None:
        if len(args.directory) != 1:
            parser.error("Render one directory at a time, or combine several with --overview")
        print(render(args.directory[0], args.note))
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_style(plt)
    load = lambda directory: [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*-r*.json"))]
    groups = [(Path(runs[0].get("checkpoint", "model")).name, runs) for runs in map(load, args.directory)]
    latency = [load(directory) for directory in args.latency] if args.latency else None
    fig = overview(plt, groups, latency, args.suite)
    if latency:
        count = sum(args.suite in (None, x["suite"]) for x in latency[0][0]["records"])
        fig.supxlabel(f"Latency: one {args.suite or ''} request repeated {count} times per worker run, {sum(r['backend'] == 'cuda' for r in latency[0])} runs per backend"
                      f" · accuracy: {len(groups[0][1][0]['records'])} frozen requests per model"
                      "\nJev latency is from the frozen, mixed requests and includes the network · successful-request latency",
                      fontsize=8, color=MUTED)
    else:
        fig.supxlabel(" | ".join(f"{model}: {len(runs[0]['records'])} requests x {len(runs)} worker runs" for model, runs in groups)
                      + "\nSame frozen inputs for every backend · successful-request latency · remote latency includes the network",
                      fontsize=8, color=MUTED)
    args.overview.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(args.overview.with_suffix(suffix), dpi=180, bbox_inches="tight")
    print(args.overview.with_suffix(".png"))


if __name__ == "__main__":
    main()
