"""Reproducible, process-isolated official comparisons. No GPU imports at import time."""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

MODES=("cuda","official-fp32","official-fp16","official")
# Numerical and accuracy gates compare against the SDK at FP32, the checkpoint's own precision.
# SDK FP16/BF16 are measured too: they are reduced-precision peers, not ground truth.
REFERENCE="official-fp32"
TIMED=("cuda","official-fp16","official")
MODELS=("laya","laya-multilingual","laya-typed-decisions")
# Accuracy may not fall significantly below the reference: one-sided exact sign test, p >= 0.05.
GATES={"probability_mae":1e-3,"probability_max":1e-2,"agreement":.995,"accuracy_sign_test_p":.05,"speedup":1.05}


def gpu_memory():
    try:
        output=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True)
        import os
        for line in output.splitlines():
            pid,memory=line.split(",",1)
            if int(pid)==os.getpid() and memory.strip().isdigit():
                return int(memory)*1024**2
    except (OSError,ValueError,subprocess.SubprocessError):
        pass
    return None  # WDDM/WSL may not expose per-process usage; never substitute allocator bytes.


def label(answer):
    if answer["type"]=="choice":
        return answer["choice"]
    if answer["type"]=="noul":
        return "true" if answer["noul"]>=.5 else "false"
    return max(answer["probabilities"],key=answer["probabilities"].get)


def probabilities(answer):
    p=answer.get("probabilities")
    return list(p.values()) if p is not None else [1-answer["noul"],answer["noul"]]


def cases(adapter):
    """Evaluation text differs from the repeated-payment development probes."""
    for length in (32,64,128,256,512,768,1024):
        if length>adapter.max_len:
            continue
        for count in (1,5,10):
            definition={"type":"choice","instructions":"Pick a topic.","criteria":["finance","software"]}
            queries={str(i):definition for i in range(count)}
            overhead=len(adapter.prepare("",queries)[0][0])
            # Verify the effective length, not the number of words in the source.
            state=" investigation"*max(0,length-overhead)
            actual=len(adapter.prepare(state,queries)[0][0])
            if actual!=length:
                raise RuntimeError(f"Length construction failed: requested {length}, actual {actual}")
            yield f"l{length}-q{count}",state,queries
    for length in (127,129,adapter.max_len-1):
        queries={"0":{"type":"choice","instructions":"Pick a topic.","criteria":["finance","software"]}}
        overhead=len(adapter.prepare("",queries)[0][0])
        yield f"boundary-l{length}"," investigation"*(length-overhead),queries
    queries={str(i):{"type":"choice","instructions":"Choose. "+"Different detail. "*i,
                    "criteria":["yes","no"]} for i in range(5)}
    yield "mixed-lengths","A support case about a payment.",queries


def long_cases(adapter,records,count=48):
    """Deterministic long inputs built by joining real fixture states, 60%-100% of the context budget.

    Repeated-token probes (see cases) trigger massive activations that amplify any FP16 rounding,
    including the SDK's own; they stay as stress diagnostics, while this suite gates the length regime.
    """
    low=int(adapter.max_len*.6)
    for i in range(count):
        target=low+(i*37)%(adapter.max_len-low)
        queries=json.loads(records[(i*7)%len(records)]["questions"])
        parts,j=[],i
        while True:
            parts.append(json.dumps(json.loads(records[j%len(records)]["state"]),ensure_ascii=False))
            state="\n".join(parts)
            tokens=max(len(v[0]) for v in adapter.prepare(state,queries))
            if tokens>=target or len(parts)>60:
                break
            j+=13
        yield f"long-{i}",state,queries,tokens


def worker(args):
    import laya_cuda
    from laya_cuda import Engine
    import psutil
    start=time.perf_counter()
    with Engine(args.checkpoint,registry=getattr(args,"registry",None),
                backend="cuda" if args.mode=="cuda" else "official") as engine:
        load_ms=(time.perf_counter()-start)*1000
        if args.mode in ("official-fp16",REFERENCE):
            engine.runtime.agent.dtype=getattr(engine.runtime.torch,"float16" if args.mode=="official-fp16" else "float32")
        result={"model":args.model,"mode":args.mode,"round":args.round,"load_ms":load_ms,
                "platform":platform.platform(),"python":platform.python_version(),"timing":{},"quality":[],
                "protocol":{"warmup":args.warmup,"samples":args.samples},
                "gates":GATES,"packages":{d.metadata["Name"]:d.version for d in metadata.distributions()
                    if d.metadata["Name"] in ("laya-cuda","cupy-cuda12x","cupy-cuda13x","numpy","laya","torch","transformers")}}
        result["checkpoint_files"]={str(p.relative_to(engine.path)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in engine.path.rglob("*") if p.is_file() and p.suffix in (".json",".safetensors")}
        result["implementation"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(laya_cuda.__file__).parent.iterdir() if p.suffix in (".py",".cu",".json")}
        result["benchmark_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        try:
            result["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip()
        except (OSError,subprocess.SubprocessError):
            result["gpu"]=None
        if not args.quality_only:
            workloads=list(cases(engine.adapter))
            requested=set(args.cases or [])
            if requested-{v[0] for v in workloads}:
                raise ValueError("Requested cases are unavailable for this model")
            for name,state,queries in workloads:
                if requested and name not in requested:
                    continue
                cold=time.perf_counter()
                output=engine.predict(state,queries)
                first_ms=(time.perf_counter()-cold)*1000
                first_metrics=engine.last_metrics.copy()
                if args.mode==REFERENCE:  # answers only: the reference is not a latency baseline
                    result["timing"][name]={"answers":output["answers"],"first_ms":first_ms,
                        "token_lengths":[len(v[0]) for v in engine.adapter.prepare(state,queries)]}
                    continue
                for _ in range(args.warmup):
                    engine.predict(state,queries)
                times,gpu=[],[]
                for _ in range(args.samples):
                    begin=time.perf_counter()
                    engine.predict(state,queries)
                    times.append((time.perf_counter()-begin)*1000)
                    if "gpu_ms" in engine.last_metrics:
                        gpu.append(engine.last_metrics["gpu_ms"])
                result["timing"][name]={"ms":times,"gpu_ms":gpu,"first_ms":first_ms,"first_metrics":first_metrics,"answers":output["answers"],
                    "p50":float(np.percentile(times,50)),"p95":float(np.percentile(times,95)),"p99":float(np.percentile(times,99)),
                    "questions_per_second":len(queries)*1000/float(np.mean(times)),"metrics":engine.last_metrics,
                    "rss_bytes":psutil.Process().memory_info().rss,"process_gpu_bytes":gpu_memory(),
                    "token_lengths":[len(v[0]) for v in engine.adapter.prepare(state,queries)]}
                print(args.model,args.mode,name,round(result["timing"][name]["p50"],3),flush=True)
        if args.dataset:
            import pyarrow.parquet as pq
            records=pq.read_table(args.dataset).to_pylist()
            result["dataset_sha256"]=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
            for row in records:
                state,queries=json.loads(row["state"]),json.loads(row["questions"])
                output=engine.predict(state,queries)
                result["quality"].append({"id":row["id"],"answers":output["answers"],"gold":json.loads(row["gold"])})
            result["long"]=[{"id":name,"tokens":tokens,"answers":engine.predict(state,queries)["answers"]}
                            for name,state,queries,tokens in long_cases(engine.adapter,records)]
        result["torch_imported"]="torch" in sys.modules
        if args.mode=="cuda" and result["torch_imported"]:
            raise RuntimeError("Lightweight benchmark worker imported Torch")
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")


def compare(candidate,reference):
    """Probability, action and label agreement of aligned answer records."""
    diffs,actions,agree,total=[],[],0,0
    for x,y in zip(candidate,reference,strict=True):
        if x["id"]!=y["id"]:
            raise ValueError("Answer records do not align")
        for q in x["answers"]:
            a,b=x["answers"][q],y["answers"][q]
            diffs.extend(np.abs(np.array(probabilities(a))-probabilities(b)).tolist())
            actions.append(abs(a["action"]["act_probability"]-b["action"]["act_probability"]))
            agree+=label(a)==label(b);total+=1
    return {"decisions":total,"mae":round(float(np.mean(diffs)),12) if diffs else None,"max":round(max(diffs),12) if diffs else None,
            "agreement":agree/total if total else None,"finite":bool(np.isfinite(diffs).all()),
            "action_mae":round(float(np.mean(actions)),12) if actions else None,"action_max":round(max(actions),12) if actions else None}


def accuracy_test(candidate,reference):
    """Exact one-sided sign test on the decisions exactly one side gets right.

    Near-tie flips move plain accuracy by a decision or two in either direction; a deficit fails
    only when chance would produce at least that many losses with probability below the gate.
    """
    wins,losses=0,0
    for x,y in zip(candidate,reference,strict=True):
        for q,gold in x["gold"].items():
            a,b=label(x["answers"][q])==str(gold["label"]),label(y["answers"][q])==str(gold["label"])
            wins+=a and not b
            losses+=b and not a
    n=wins+losses
    return {"wins":wins,"losses":losses,"p_value":sum(math.comb(n,k) for k in range(losses,n+1))/2**n if n else 1.0}


def passes(stats):
    return bool(stats["decisions"] and stats["finite"] and stats["mae"]<=GATES["probability_mae"] and stats["max"]<=GATES["probability_max"]
        and stats["action_mae"]<=GATES["probability_mae"] and stats["action_max"]<=GATES["probability_max"]
        and stats["agreement"]>=GATES["agreement"])


def summarize(directory):
    runs=[json.loads(p.read_text(encoding="utf-8")) for p in Path(directory).glob("*-r*.json")]
    summary={"gates":GATES,"reference":REFERENCE,"models":{}}
    for model in sorted({r["model"] for r in runs}):
        modes={mode:sorted([r for r in runs if r["model"]==model and r["mode"]==mode],key=lambda r:r["round"]) for mode in MODES}
        if not all(modes.values()):
            continue
        first={mode:group[0] for mode,group in modes.items()}
        c,ref=first["cuda"],first[REFERENCE]
        quality=compare(c["quality"],ref["quality"])
        correct={mode:sum(label(a)==str(row["gold"][q]["label"]) for row in run["quality"] for q,a in row["answers"].items())
                 for mode,run in first.items()}
        quality["accuracy"]={mode:v/quality["decisions"] for mode,v in correct.items()} if quality["decisions"] else {}
        quality["accuracy_test"]=accuracy_test(c["quality"],ref["quality"])
        quality["passed"]=passes(quality) and quality["accuracy_test"]["p_value"]>=GATES["accuracy_sign_test_p"]
        long=None
        if all("long" in run for run in first.values()):
            long=compare(c["long"],ref["long"])
            long["tokens"]=[min(v["tokens"] for v in ref["long"]),max(v["tokens"] for v in ref["long"])]
            long["passed"]=passes(long)
            # Reduced-precision SDK paths on the same inputs, for context only.
            long["peers"]={mode:compare(first[mode]["long"],ref["long"]) for mode in ("official-fp16","official")}
        complete=all(len(modes[mode])==3 and {r["round"] for r in modes[mode]}=={0,1,2}
            and all(r.get("protocol",{}).get("warmup",0)>=20 and r.get("protocol",{}).get("samples",0)>=200 for r in modes[mode])
            for mode in TIMED)
        timing={}
        for key in c["timing"]:
            record=lambda run: [{"id":key,"answers":run["timing"][key]["answers"]}]
            stress=compare(record(c),record(ref))
            peer=compare(record(first["official-fp16"]),record(ref))
            comparisons=[]
            for candidate,official in zip(modes["cuda"],modes["official"],strict=True):
                a,b=candidate["timing"][key],official["timing"][key]
                comparisons.append({"speedup":b["p50"]/a["p50"],"p95_ratio":a["p95"]/b["p95"],"p99_ratio":a["p99"]/b["p99"]})
            timing[key]={"rounds":comparisons,"probability_mae":stress["mae"],"probability_max":stress["max"],
                "labels_match_reference":stress["agreement"]==1,"official_fp16_probability_max":peer["max"],
                "verified_speedup":quality["passed"] and bool(long and long["passed"]) and stress["finite"] and complete
                and all(len(r["timing"][key]["ms"])>=200 for mode in TIMED for r in modes[mode])
                and all(v["speedup"]>=1/.95 and v["p95_ratio"]<=1.05 and v["p99_ratio"]<=1.05 for v in comparisons)}
        summary["models"][model]={"quality":quality,"long":long,"timing":timing}
    Path(directory,"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir",default="models")
    parser.add_argument("--models",nargs="+",default=list(MODELS),help="Built-in or custom registry aliases")
    parser.add_argument("--registry",help="External model registry JSON; aliases take precedence over --models-dir")
    parser.add_argument("--cases",nargs="+",help="Run only these workload IDs")
    parser.add_argument("--output",default="reports/benchmark")
    parser.add_argument("--dataset")
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--samples",type=int,default=200)
    parser.add_argument("--quality-only",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--model",default="laya")
    parser.add_argument("--mode",choices=MODES,default="cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--round",type=int,default=0)
    args=parser.parse_args()
    if any(not model or model in (".","..") or any(c in model for c in '/\\:') for model in args.models):
        parser.error("Use model aliases, not paths, in --models; configure paths in --registry")
    if any(importlib.util.find_spec(name) is None for name in ("psutil","laya","torch")) or (args.dataset and importlib.util.find_spec("pyarrow") is None):
        parser.error('Install "laya-cuda[benchmark]" to run official comparisons')
    if min(args.rounds,args.samples)<1 or args.warmup<0:
        parser.error("rounds and samples must be positive; warmup must be nonnegative")
    if args.worker:
        return worker(args)
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    if any(target.glob("*-r*.json")):
        parser.error("Output contains existing runs; choose a fresh --output directory")
    for model in args.models:
        for r in range(args.rounds):
            for mode in (MODES if r%2==0 else MODES[::-1]):
                if mode==REFERENCE and r:
                    continue  # answers are deterministic; one reference run serves every round
                output=target/f"{model}-{mode}-r{r}.json"
                command=[sys.executable,"-m","benchmarks.run","--worker","--model",model,"--mode",mode,
                    "--checkpoint",model if args.registry else str(Path(args.models_dir)/model),
                    "--round",str(r),"--output",str(output),
                    "--warmup",str(args.warmup),"--samples",str(args.samples)]
                if args.registry:
                    command += ["--registry",str(Path(args.registry).resolve())]
                if args.dataset and r==0:
                    command += ["--dataset",args.dataset]
                if args.cases:
                    command += ["--cases",*args.cases]
                if args.quality_only:
                    command.append("--quality-only")
                subprocess.run(command,check=True)
    summary=summarize(target)
    print(json.dumps({model:values["quality"] for model,values in summary["models"].items()},indent=2))


if __name__=="__main__":
    main()
