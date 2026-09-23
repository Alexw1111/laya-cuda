"""Diagnostic layer parity against the actual SDK, outside timed benchmarks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cupy as cp
import numpy as np
from laya_cuda import Engine

parser=argparse.ArgumentParser()
parser.add_argument("--output",required=True)
args=parser.parse_args()
results={}
queries={"a":{"type":"choice","instructions":"Which team?","criteria":["billing","technical"]},
         "b":{"type":"noul","instructions":"Is this urgent?"}}
for model in ("laya","laya-multilingual","laya-typed-decisions"):
    results[model]={}
    with Engine(Path("models")/model) as candidate, Engine(Path("models")/model,backend="official") as reference:
        torch=reference.runtime.torch
        reference.runtime.agent.dtype=torch.float16
        expected={}
        def save(name):
            def hook(module,inputs,output):
                expected[name]=(output[0] if isinstance(output,tuple) else output).detach().float().cpu().numpy()
            return hook
        encoder=reference.runtime.agent.model.encoder
        handles=[layer.register_forward_hook(save(f"layer-{i}")) for i,layer in enumerate(encoder.layers)]
        handles.append(encoder.final_norm.register_forward_hook(save("final_norm")))
        try:
            for regime,state in (("short","Please refund my payment."),("long","Please refund my payment. "*1000)):
                expected.clear()
                official=reference.predict(state,queries)
                items=candidate.adapter.prepare(state,queries)
                actual=candidate.predict(state,queries)
                slot=next(reversed(candidate.runtime.slots.values()))
                observed={}
                original=slot.norm
                def trace(target,prefix=None,branch=None,source=None,eps=None):
                    if branch is not None and prefix and (prefix=="encoder.final_norm" or prefix.endswith(".attn_norm")):
                        index=candidate.adapter.layers-1 if prefix=="encoder.final_norm" else int(prefix.split(".")[2])-1
                        observed[f"layer-{index}"]=cp.asnumpy(slot.residual+branch.astype(cp.float32))
                    original(target,prefix,branch,source,eps)
                    if prefix=="encoder.final_norm":
                        observed["final_norm"]=cp.asnumpy(slot.residual)
                slot.norm=trace
                try:
                    with candidate.runtime.device,cp.cuda.using_allocator(candidate.runtime.pool.malloc),candidate.runtime.stream:
                        slot.fill(items)
                        slot.encoder()
                        candidate.runtime.stream.synchronize()
                finally:
                    del slot.norm
                layers={}
                for name,oracle in expected.items():
                    values=observed[name].reshape(slot.b,slot.l,-1)
                    differences=np.concatenate([np.abs(values[i,:len(item[0])]-oracle[i,:len(item[0])]).ravel() for i,item in enumerate(items)])
                    assert np.isfinite(differences).all()
                    layers[name]={"mae":float(differences.mean()),"max":float(differences.max())}
                results[model][regime]={"layers":layers,"candidate":actual,"official_fp16":official}
                print(model,regime,"final_norm",layers["final_norm"],flush=True)
        finally:
            for handle in handles:
                handle.remove()
Path(args.output).write_text(json.dumps(results,indent=2),encoding="utf-8")
