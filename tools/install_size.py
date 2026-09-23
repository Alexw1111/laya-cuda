"""Report resolved PyPI wheel bytes for a clean installation, excluding model weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata as metadata
import json
from pathlib import Path
import urllib.request

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

parser=argparse.ArgumentParser()
parser.add_argument("--output",required=True)
args=parser.parse_args()
cached=json.loads(Path(args.output).read_text())["packages"] if Path(args.output).exists() else {}
tags=list(sys_tags())
rank={tag:i for i,tag in enumerate(tags)}

def size(dist):
    name,version=dist.metadata["Name"],dist.version
    if name in cached and cached[name]["version"]==version and "error" not in cached[name]:
        return name,cached[name]
    if name.lower() in ("pip","laya-cuda"):
        return name,{"version":version,"bytes":None,"note":"seed tool or locally built wheel"}
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json",timeout=30) as response:
            info=json.load(response)
        candidates=[]
        for f in info["urls"]:
            if f["packagetype"]!="bdist_wheel":
                continue
            _,_,_,wheel_tags=parse_wheel_filename(f["filename"])
            matches=[rank[t] for t in wheel_tags if t in rank]
            if matches:
                candidates.append((min(matches),f))
        f=min(candidates,key=lambda x:x[0])[1]
        return name,{"version":version,"bytes":f["size"],"filename":f["filename"],"sha256":f["digests"]["sha256"]}
    except Exception as error:
        return name,{"version":version,"bytes":None,"error":str(error)}

with ThreadPoolExecutor(max_workers=4) as pool:
    entries=dict(pool.map(size,metadata.distributions()))
result={"packages":entries,"compressed_dependency_bytes":sum(v["bytes"] or 0 for v in entries.values()),
        "complete":not any("error" in v for v in entries.values()),
        "note":"PyPI wheel metadata for installed versions/platform. Excludes pip, local laya-cuda wheel, models, caches and HTTP overhead."}
Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
print(result["compressed_dependency_bytes"],result["complete"])
