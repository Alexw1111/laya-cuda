"""Run with an installed wheel using python -I; no source-tree imports allowed."""
import argparse
import ctypes
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

parser=argparse.ArgumentParser()
parser.add_argument("--models-dir",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--full",action="store_true")
parser.add_argument("--expected-cuda-major",type=int)
args=parser.parse_args()
for key in list(os.environ):
    if key.startswith(("CUDA_PATH","CUDA_HOME")) or key in ("NVCC","CUDACXX","LD_LIBRARY_PATH"):
        os.environ.pop(key)
os.environ["PATH"]=os.pathsep.join([str(Path(sys.executable).parent),r"C:\Windows\System32"]
    if os.name=="nt" else [str(Path(sys.executable).parent),"/usr/bin","/bin"])
os.environ["CUDA_PATH"]=str(Path(sys.prefix)/"no-system-toolkit")
cache=tempfile.TemporaryDirectory(prefix="laya-cuda-cold-kernels-")
os.environ["CUPY_CACHE_DIR"]=cache.name
packages={d.metadata["Name"].lower():d.version for d in metadata.distributions()}
if not args.full:
    assert not {"torch","transformers","tensorrt"}&packages.keys(),packages
started=time.perf_counter()
import laya_cuda
assert Path(laya_cuda.__file__).is_relative_to(Path(sys.prefix)),laya_cuda.__file__
assert not {"torch","transformers","cupy"}&sys.modules.keys()
result={"platform":platform.platform(),"python":platform.python_version(),"packages":packages,
        "import_ms":(time.perf_counter()-started)*1000,"package_path":laya_cuda.__file__,"models":{}}
queries={"choice":{"type":"choice","instructions":"Pick a topic.","criteria":["payment","software"]},
         "score":{"type":"score","instructions":"Urgency?","criteria":["low","high"]},
         "noul":{"type":"noul","instructions":"Urgent?"}}
for name in ("laya","laya-multilingual","laya-typed-decisions"):
    started=time.perf_counter()
    with laya_cuda.Engine(Path(args.models_dir)/name) as engine:
        load=(time.perf_counter()-started)*1000
        output=engine.predict("Please refund my payment.",queries)
        first=engine.last_metrics.copy()
        assert output==engine.predict("Please refund my payment.",queries)
        engine.predict("Please refund my payment. "*1000,queries)
        result["models"][name]={"load_ms":load,"first":first,"long":engine.last_metrics,"answers":output["answers"]}
        if "cuda_runtime_version" not in result:
            import cupy as cp
            result["cuda_runtime_version"] = cp.cuda.runtime.runtimeGetVersion()
            result["nvrtc_version"] = list(cp.cuda.nvrtc.getVersion())
            if args.expected_cuda_major is not None:
                assert result["cuda_runtime_version"]//1000 == args.expected_cuda_major, result["cuda_runtime_version"]
                assert result["nvrtc_version"][0] == args.expected_cuda_major, result["nvrtc_version"]
    assert engine.runtime.pool.used_bytes()==0
assert not {"torch","transformers"}&sys.modules.keys()
if os.name=="nt":
    from ctypes import wintypes
    psapi=ctypes.WinDLL("psapi")
    process=ctypes.windll.kernel32.GetCurrentProcess()
    handles,needed=(ctypes.c_void_p*4096)(),wintypes.DWORD()
    psapi.EnumProcessModules(ctypes.c_void_p(process),handles,ctypes.sizeof(handles),ctypes.byref(needed))
    loaded=[]
    for handle in handles[:needed.value//ctypes.sizeof(ctypes.c_void_p)]:
        name=ctypes.create_unicode_buffer(32768)
        psapi.GetModuleFileNameExW(ctypes.c_void_p(process),ctypes.c_void_p(handle),name,len(name))
        if Path(name.value).name.lower().startswith(("nvrtc","cublas","cudart")):
            loaded.append(name.value)
else:
    loaded=sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if any(s in line for s in ("libnvrtc","libcublas","libcudart"))})
assert loaded and all(str(Path(sys.prefix)) in p for p in loaded),loaded
result["loaded_cuda_libraries"]=loaded
result["installed_bytes"]=sum(p.stat().st_size for p in Path(sys.prefix).rglob("*") if p.is_file())
result["core_passed"]=True
if args.full:
    try:
        with laya_cuda.Engine(Path(args.models_dir)/"laya",backend="official") as engine:
            engine.predict("Please refund my payment.",queries)
        result["official_passed"]=True
    except (ImportError,RuntimeError) as error:
        result["official_passed"]=False
        result["official_error"]=str(error)
Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
cache.cleanup()
print(json.dumps({k:v for k,v in result.items() if k not in ("models","loaded_cuda_libraries","packages")},indent=2))
