"""Smoke-test the installed core wheel with a real worker-owned CUDA engine."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

packages={d.metadata['Name'].lower() for d in metadata.distributions()}
assert not {'torch','transformers','tensorrt'}&packages
from laya_cuda import BatchEngine
import laya_cuda
assert Path(laya_cuda.__file__).is_relative_to(Path(sys.prefix))
assert not {'cupy','torch','transformers'}&sys.modules.keys()
questions={'q':{'type':'choice','instructions':'Choose a topic.','criteria':['payment','software']}}
with BatchEngine(sys.argv[1]) as engine:
    futures=[engine.submit('Please refund my payment.',questions) for _ in range(8)]
    results=[f.result() for f in futures]
    assert all(result==results[0] for result in results)
assert not engine._thread.is_alive() and engine._pending==0
assert not {'torch','transformers','tensorrt'}&sys.modules.keys()
print(json.dumps({'installed_package':laya_cuda.__file__,'requests':len(results),'worker_closed':True,'heavy_imports':False}))
