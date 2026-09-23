"""Attribute warmed graph time to operations; instrumentation adds overhead."""
import argparse
import ctypes as ct
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cupy as cp
import numpy as np
from cuda.pathfinder import load_nvidia_dynamic_lib
from laya_cuda import Engine

parser=argparse.ArgumentParser()
parser.add_argument("--model",default="laya")
parser.add_argument("--length",type=int,default=512)
parser.add_argument("--questions",type=int,default=10)
parser.add_argument("--output",required=True)
args=parser.parse_args()
record=ct.CDLL(load_nvidia_dynamic_lib("cudart").abs_path).cudaEventRecordWithFlags
record.argtypes,record.restype=[ct.c_void_p,ct.c_void_p,ct.c_uint],ct.c_int
with Engine(Path("models")/args.model) as engine:
    queries={str(i):{"type":"choice","instructions":"Choose a department.","criteria":["billing","technical"]} for i in range(args.questions)}
    overhead=len(engine.adapter.prepare("",queries)[0][0])
    state=" refund"*(args.length-overhead)
    engine.predict(state,queries)
    runtime=engine.runtime
    slot=next(reversed(runtime.slots.values()))
    events=[]
    stage=["encoder"]
    busy=[False]
    def wrap(fn,label):
        def timed(*a,**kw):
            if busy[0]:
                return fn(*a,**kw)
            busy[0]=True
            start,end=cp.cuda.Event(),cp.cuda.Event()
            key=stage[0]+"/"+(str(a[0]) if label=="kernel" else "gemm")
            assert record(start.ptr,runtime.stream.ptr,1)==0  # cudaEventRecordExternal
            result=fn(*a,**kw)
            assert record(end.ptr,runtime.stream.ptr,1)==0
            events.append((key,start,end))
            busy[0]=False
            return result
        return timed
    original=engine.adapter.forward_head
    def head(s):
        stage[0]="head"
        original(s)
        stage[0]="encoder"
    engine.adapter.forward_head=head
    for name in ("linear","call"):
        setattr(runtime.ops,name,wrap(getattr(runtime.ops,name),"kernel" if name=="call" else "gemm"))
    with runtime.stream:
        runtime.stream.begin_capture()
        slot.forward()
        graph=runtime.stream.end_capture()
        for _ in range(20):graph.launch(runtime.stream)
        runtime.stream.synchronize()
        samples=defaultdict(list)
        for _ in range(30):
            graph.launch(runtime.stream)
            runtime.stream.synchronize()
            totals=defaultdict(float)
            for key,start,end in events:totals[key]+=cp.cuda.get_elapsed_time(start,end)
            for key,value in totals.items():samples[key].append(value)
    result={"model":args.model,"shape":engine.last_metrics["shape"],"instrumented":True,
            "ms":{k:float(np.median(v)) for k,v in sorted(samples.items())}}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
