"""Small CUDA execution primitives; cuBLAS calls also work inside graph capture."""
import ctypes as ct
from pathlib import Path

import cupy as cp
from cuda.pathfinder import load_nvidia_dynamic_lib
import numpy as np


class Ops:
    def __init__(self, stream):
        self.stream = stream
        self.module = cp.RawModule(code=Path(__file__).with_name("kernels.cu").read_text(),
                                   options=("--std=c++17", "--fmad=false"))
        self.kernels = {n: self.module.get_function("layer_norm" if n == "norm" else n) for n in (
            "embed", "norm", "bias", "add", "activate", "layout", "attention",
            "type_add", "gather", "gather_half", "features")}
        self.blas = ct.CDLL(load_nvidia_dynamic_lib("cublas").abs_path)
        p, i = ct.c_void_p, ct.c_int
        for name, args in {
            "cublasCreate_v2": [ct.POINTER(p)], "cublasDestroy_v2": [p],
            "cublasSetStream_v2": [p, p],
            "cublasSetMathMode": [p, i],
            "cublasGemmEx": [p,i,i,i,i,i,p,p,i,i,p,i,i,p,p,i,i,i,i],
        }.items():
            function = getattr(self.blas, name)
            function.argtypes, function.restype = args, i
        self.handle, self.alpha, self.beta = p(), ct.c_float(1), ct.c_float(0)
        self.check(self.blas.cublasCreate_v2(ct.byref(self.handle)))
        self.check(self.blas.cublasSetStream_v2(self.handle, stream.ptr))
        # Keep intermediate reductions in FP32, even when GEMM output is FP16.
        self.check(self.blas.cublasSetMathMode(self.handle, 16))

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"cuBLAS failed with status {code}")

    def call(self, name, blocks, *args, threads=256):
        converted = tuple(np.int32(a) if isinstance(a, int) else np.float32(a) if isinstance(a, float)
                          else np.uint64(0) if a is None else a for a in args)
        self.kernels[name]((blocks,), (threads,), converted, stream=self.stream)

    def elements(self, name, n, *args):
        self.call(name, (n+255)//256, *args)

    def linear(self, x, weight, out, bias=None, scratch=None):
        m, k, n = x.size // weight.shape[1], weight.shape[1], weight.shape[0]
        target = scratch if bias is not None else out
        if target is None or target.size<m*n:
            raise ValueError("Biased GEMM requires an FP32 scratch buffer")
        self.check(self.blas.cublasGemmEx(self.handle, 1, 0, n, m, k, ct.byref(self.alpha),
            weight.data.ptr, 2, k, x.data.ptr, 2, k, ct.byref(self.beta), target.data.ptr, 0 if target.dtype == cp.float32 else 2, n, 68, -1))
        if bias is not None:
            self.elements("bias", m*n, target, bias, out, m*n, n, int(out.dtype == cp.float32))

    def close(self):
        if self.handle:
            self.check(self.blas.cublasDestroy_v2(self.handle))
            self.handle = ct.c_void_p()
