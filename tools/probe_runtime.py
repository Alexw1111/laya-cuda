"""Verify wheel-only CUDA primitives with system toolkits hidden from discovery."""
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import time

# Test-process isolation only: do not modify the user's global environment.
for key in list(os.environ):
    if key.startswith(("CUDA_PATH", "CUDA_HOME")) or key in ("NVCC", "CUDACXX", "LD_LIBRARY_PATH"):
        os.environ.pop(key)
os.environ["PATH"] = os.pathsep.join(
    [str(Path(sys.executable).parent), r"C:\Windows\System32"]
    if os.name == "nt" else [str(Path(sys.executable).parent), "/usr/bin", "/bin"]
)
os.environ["CUDA_PATH"] = str(Path(sys.prefix) / "no-system-toolkit")

import numpy as np
from cuda.pathfinder import load_nvidia_dynamic_lib

started = time.perf_counter()
import cupy as cp

libraries = {}
for name in ("cudart", "nvrtc", "cublas"):
    lib = load_nvidia_dynamic_lib(name)
    libraries[name] = str(lib)

source = r'''
#include <cuda_fp16.h>
extern "C" __global__ void twice(const half* x, half* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(2.0f * __half2float(x[i]));
}
'''
kernel = cp.RawKernel(source, "twice", options=("--std=c++17",))
x = cp.asarray(np.arange(256, dtype=np.float16))
y = cp.empty_like(x)
kernel((1,), (256,), (x, y, np.int32(x.size)))
np.testing.assert_array_equal(y.get(), np.arange(256, dtype=np.float16) * 2)
a, b = cp.ones((64, 64), dtype=cp.float16), cp.ones((64, 64), dtype=cp.float16)
c = cp.empty_like(a)
cp.matmul(a, b, out=c)
np.testing.assert_array_equal(c.get(), np.full((64, 64), 64, dtype=np.float16))
stream = cp.cuda.Stream(non_blocking=True)
# CuPy 14.2's wrapper rejects cuBLAS calls during capture. Exercise the native
# cuBLAS API directly; this is a real integration cost, not a system setup step.
blas = ctypes.CDLL(load_nvidia_dynamic_lib("cublas").abs_path)
p, i = ctypes.c_void_p, ctypes.c_int
blas.cublasCreate_v2.argtypes = [ctypes.POINTER(p)]
blas.cublasSetStream_v2.argtypes = [p, p]
blas.cublasDestroy_v2.argtypes = [p]
blas.cublasGemmEx.argtypes = [p, i, i, i, i, i, p, p, i, i, p, i, i, p, p, i, i, i, i]
handle = p()
assert blas.cublasCreate_v2(ctypes.byref(handle)) == 0
assert blas.cublasSetStream_v2(handle, stream.ptr) == 0
alpha, beta = ctypes.c_float(1), ctypes.c_float(0)

def gemm():
    status = blas.cublasGemmEx(handle, 0, 0, 64, 64, 64, ctypes.byref(alpha),
                               a.data.ptr, 2, 64, b.data.ptr, 2, 64, ctypes.byref(beta),
                               c.data.ptr, 2, 64, 68, -1)
    assert status == 0, status

with stream:
    gemm()
stream.synchronize()
with stream:
    stream.begin_capture()
    kernel((1,), (256,), (x, y, np.int32(x.size)))
    gemm()
    graph = stream.end_capture()
graph.launch(stream)
stream.synchronize()
np.testing.assert_array_equal(c.get(), np.full((64, 64), 64, dtype=np.float16))
assert blas.cublasDestroy_v2(handle) == 0

if os.name == "nt":
    # Record the actual loaded libraries, including any accidental system-toolkit use.
    from ctypes import wintypes
    psapi = ctypes.WinDLL("psapi")
    process = ctypes.windll.kernel32.GetCurrentProcess()
    handles, needed = (ctypes.c_void_p * 4096)(), wintypes.DWORD()
    psapi.EnumProcessModules(ctypes.c_void_p(process), handles, ctypes.sizeof(handles), ctypes.byref(needed))
    loaded = []
    for handle in handles[:needed.value // ctypes.sizeof(ctypes.c_void_p)]:
        name = ctypes.create_unicode_buffer(32768)
        psapi.GetModuleFileNameExW(ctypes.c_void_p(process), ctypes.c_void_p(handle), name, len(name))
        if Path(name.value).name.lower().startswith(("nvrtc", "cublas", "cudart")):
            loaded.append(name.value)
else:
    loaded = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if any(s in line for s in ("libnvrtc", "libcublas", "libcudart"))})
assert loaded, "Could not verify actual CUDA library origins"
assert all(str(Path(sys.prefix)) in path for path in loaded), loaded
assert "torch" not in sys.modules and "transformers" not in sys.modules
print(json.dumps({"platform": platform.platform(), "python": platform.python_version(),
                  "cupy": cp.__version__, "runtime": cp.cuda.runtime.runtimeGetVersion(),
                  "driver": cp.cuda.runtime.driverGetVersion(), "loaded_libraries": loaded,
                  "process_seconds": time.perf_counter() - started,
                  "kernel_cache": "existing cache allowed; not a cold-compile benchmark",
                  "passed": ["nvrtc_fp16", "cublas_fp16", "cuda_graph", "wheel_libraries_only", "no_torch"]}, indent=2))
