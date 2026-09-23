"""Compare the previous wheel, current CUDA path and real SDK in fresh processes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.run import summarize

parser=argparse.ArgumentParser()
parser.add_argument("--baseline-wheel",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--cases",nargs="+",default=["l32-q1","l512-q1","l512-q10"])
args=parser.parse_args()
output=Path(args.output).resolve()
output.mkdir(parents=True,exist_ok=True)
if any(output.glob("*-r*.json")):
    parser.error("Use a fresh output directory")
wheel=Path(args.baseline_wheel).resolve()
with tempfile.TemporaryDirectory(prefix="laya-cuda-baseline-") as temp:
    baseline=Path(temp)
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            path=Path(name)
            if name.startswith("laya_cuda/") and ".." not in path.parts and not name.endswith("/"):
                target=baseline/path
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read(name))
    # Otherwise "before" would silently import the current package.
    if not (baseline/"laya_cuda").is_dir():
        parser.error("The baseline wheel has no laya_cuda package (built before the rename?)")
    # One external harness measures both implementations without modifying either.
    invocation="import sys;sys.path.insert(0,sys.argv.pop(1));from benchmarks.run import main;main()"
    for r in range(3):
        modes=("before","cuda","official-fp32","official-fp16","official")
        for mode in modes if r%2==0 else modes[::-1]:
            if mode=="official-fp32" and r:
                continue
            destination=output/f"laya-{mode}-r{r}.json"
            command=[sys.executable,"-c",invocation,str(baseline if mode=="before" else ROOT),
                "--worker","--model","laya","--mode","cuda" if mode=="before" else mode,
                "--checkpoint",str(ROOT/"models/laya"),"--round",str(r),"--output",str(destination),
                "--warmup","20","--samples","200","--cases",*args.cases]
            if r==0:
                command += ["--dataset",str(ROOT/"benchmarks/data/typed-decisions.parquet")]
            subprocess.run(command,cwd=ROOT,check=True)
            if mode=="before":
                record=json.loads(destination.read_text(encoding="utf-8"))
                record.update(mode="before",baseline_wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
                destination.write_text(json.dumps(record,indent=2),encoding="utf-8")
    summarize(output)
    comparisons={}
    for case in args.cases:
        rounds=[]
        for r in range(3):
            before=json.loads((output/f"laya-before-r{r}.json").read_text(encoding="utf-8"))["timing"][case]
            after=json.loads((output/f"laya-cuda-r{r}.json").read_text(encoding="utf-8"))["timing"][case]
            rounds.append({"p50_before":before["p50"],"p50_after":after["p50"],
                "p50_reduction":1-after["p50"]/before["p50"],
                "p95_ratio":after["p95"]/before["p95"],"p99_ratio":after["p99"]/before["p99"]})
        comparisons[case]=rounds
    (output/"before-after.json").write_text(json.dumps(comparisons,indent=2),encoding="utf-8")
    print(json.dumps(comparisons,indent=2))
