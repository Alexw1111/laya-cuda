"""Freeze task comparisons and replay verified inputs from a source snapshot."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_files(path):
    from laya_cuda.models import FILES
    return {p.relative_to(path).as_posix(): digest(p)
            for pattern in FILES for p in sorted(path.glob(pattern)) if p.is_file()}


def environment():
    def command(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "uv": command(["uv", "--version"]),
            "gpu": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv,noheader"]),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))}


def freeze(args):
    bundle = args.output / "reproduce"
    project = bundle / "project"
    project.mkdir(parents=True)
    for name in ("laya_cuda", "benchmarks"):
        shutil.copytree(ROOT/name, project/name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "uv.lock", ".python-version", "README.md", "LICENSE"):
        shutil.copy2(ROOT/name, project/name)
    shutil.copy2(args.cases_file, bundle/"inputs.jsonl")
    origin = args.cases_file.with_suffix(".manifest.json")
    if origin.exists():
        shutil.copy2(origin, bundle/"inputs.manifest.json")
    source = args.checkpoint_source or (args.checkpoint if not Path(args.checkpoint).is_dir() else None)
    checkpoint = None
    if any(b != "jev" for b in args.backends):
        from laya_cuda.models import resolve
        checkpoint = {"source": source, "files": checkpoint_files(resolve(args.checkpoint))}
    manifest = {"schema": 1, "checkpoint": checkpoint, "environment": environment(),
                "protocol": {k: getattr(args, k) for k in ("backends", "rounds", "warmup", "limit", "jev_model", "max_cost")},
                "files": {p.relative_to(bundle).as_posix(): digest(p) for p in sorted(bundle.rglob("*")) if p.is_file()}}
    (bundle/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle/"REPRODUCE.md").write_text(
        "# Reproduce this comparison\n\nCopy the entire result directory; weights and credentials are excluded.\n"
        "From this folder:\n\n```sh\ncd project\n"
        f"uv sync --locked --extra benchmark --python {platform.python_version()}\n"
        "uv run --no-sync python -m benchmarks.reproduce --bundle .. --output ../../replayed\n```\n\n"
        "Use `--checkpoint /path/to/identical/weights` for private/local checkpoints. "
        "A recorded public alias downloads its pinned revision. All consumed model files must match. "
        "Upstream is cloned at the recorded revision when needed. Jev runs require a key and paid calls. "
        "A remote model alias can change; returned model IDs are recorded but cannot guarantee bitwise replay.\n\n"
        "The same input, source and model bytes are enforced. Hardware, driver and platform differences "
        "are recorded; latency and floating-point outputs are not guaranteed identical. "
        "Close competing GPU workloads. Warmup does not warm every shape; cache misses remain measured.\n",
        encoding="utf-8")
    return bundle


def verify(bundle):
    manifest = json.loads((bundle/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported reproduction manifest")
    for name, expected in manifest["files"].items():
        path = (bundle/name).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Reproduction file mismatch: {name}")
    return manifest


def pack(directory):
    """Export an allowlisted bundle; never traverse environments or weight caches."""
    directory = Path(directory)
    bundle = directory/"reproduce"
    manifest = verify(bundle)
    files = [bundle/name for name in manifest["files"]]
    files += [bundle/"manifest.json", bundle/"REPRODUCE.md"]
    for pattern in ("*-r*.json", "summary.json", "experiment.json", "report.md", "metrics.csv", "cache-diagnostics.csv",
                    "overview.png", "overview.svg", "latency.png", "latency.svg", "quality.png", "quality.svg", "resources.png", "resources.svg", "report-note.txt",
                    "VALIDATION.md", "replay-validation.json"):
        files += list(directory.glob(pattern))
    target = directory.with_name(directory.name+".zip")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(directory).as_posix())
    target.with_suffix(".zip.sha256").write_text(digest(target)+"  "+target.name+"\n", encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = verify(bundle)
    if args.verify_only:
        print("Source snapshot, lockfile and frozen inputs verified (model/environment not checked).")
        return
    if not args.output:
        parser.error("--output is required for replay")
    # Running from the snapshot prevents a later checkout from changing the experiment.
    if ROOT != bundle/"project":
        raise ValueError("Run from reproduce/project using its locked uv environment; see REPRODUCE.md")
    checkpoint = args.checkpoint or (manifest["checkpoint"] or {}).get("source")
    if manifest["checkpoint"]:
        if not checkpoint:
            parser.error("Supply --checkpoint with the original model files")
        from laya_cuda.models import resolve
        path = resolve(checkpoint)
        if checkpoint_files(path) != manifest["checkpoint"]["files"]:
            raise ValueError("Checkpoint bytes differ, including tokenizer/configuration; replay refused")
        checkpoint = str(path)
    from .backends import UPSTREAM_REVISION, upstream_source
    upstream = (args.upstream or ROOT/"vendor/laya-upstream").resolve()
    if "upstream" in manifest["protocol"]["backends"]:
        if not upstream.exists():
            upstream.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "https://github.com/nandhakishorm/laya.git", str(upstream)], check=True)
            subprocess.run(["git", "-C", str(upstream), "checkout", UPSTREAM_REVISION], check=True)
        upstream_source(upstream)
    from .compare import main as compare
    protocol = manifest["protocol"]
    command = ["compare", "--cases-file", str(bundle/"inputs.jsonl"), "--output", str(args.output.resolve()),
               "--backends", *protocol["backends"], "--checkpoint", checkpoint or "laya", "--upstream", str(upstream)]
    for key in ("rounds", "warmup", "limit", "jev_model", "max_cost"):
        if protocol[key] is not None:
            command += ["--"+key.replace("_", "-"), str(protocol[key])]
    if (manifest["checkpoint"] or {}).get("source"):
        command += ["--checkpoint-source", manifest["checkpoint"]["source"]]
    sys.argv = command
    compare()


if __name__ == "__main__":
    main()
