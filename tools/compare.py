"""Development comparison, preserving each backend's complete returned outputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from laya_cuda import Engine

parser=argparse.ArgumentParser()
parser.add_argument("--backend",default="cuda",choices=["cuda","official"])
parser.add_argument("--fp16",action="store_true")
parser.add_argument("--output",required=True)
args=parser.parse_args()
queries={"team":{"type":"choice","instructions":"Which team?","criteria":["billing","technical"]},
         "urgent":{"type":"noul","instructions":"Is this urgent?"},
         "score":{"type":"score","instructions":"Urgency","criteria":["low","medium","high"]}}
result={}
for name in ("laya","laya-multilingual","laya-typed-decisions"):
    with Engine(Path("models")/name,backend=args.backend) as e:
        if args.fp16:
            e.runtime.agent.dtype=e.runtime.torch.float16
        result[name]=[]
        for n in (2,200,2000):
            output=e.predict("Please refund my payment. "*n,queries)
            result[name].append({"output":output,"metrics":e.last_metrics})
            print(name,n,e.last_metrics,flush=True)
Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
