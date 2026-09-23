"""Resident weights and bounded shape slots for the shared encoder execution."""
from collections import OrderedDict
import warnings

with warnings.catch_warnings():
    # CUDA ships as wheels, so CuPy's system-toolkit lookup warning would only mislead users.
    warnings.filterwarnings("ignore", message="CUDA path could not be detected")
    import cupy as cp
import numpy as np
from safetensors import safe_open

from .ops import Ops


def span(tokens):
    """Rows start at multiples of 16 tokens, the attention tile."""
    return -(-tokens//16)*16


def classes(items):
    """Shape classes: packed token rows, then question and option counts rounded to powers of two."""
    rows = sum(span(len(v[0])) for v in items)
    step = 32 if rows <= 256 else 64 if rows <= 512 else 128
    power = lambda n, low: max(low, 1 << (n-1).bit_length())
    return -(-rows//step)*step, power(len(items), 1), power(max(len(v[1]) for v in items), 8)


class Slot:
    """Buffers and one CUDA graph for a shape class. Questions are packed back to back as rows."""
    def __init__(self, runtime, rows, batch, options):
        self.runtime, self.a, self.op = runtime, runtime.adapter, runtime.ops
        self.m, self.b, self.k = rows, batch, options
        self.graph = None
        a, m, b, k, w = self.a, rows, batch, options, self.a.width
        fields = (("ids", m), ("positions", m), ("rows", m), ("kinds", m), ("starts", b), ("lengths", b),
                  ("counts", b), ("indices", b*(k+1)))
        self.host = np.zeros(sum(size for _, size in fields), np.int32)
        self.device = cp.zeros_like(cp.asarray(self.host))
        self.fields, self.inputs, offset = {}, {}, 0
        for name, size in fields:
            self.fields[name] = self.host[offset:offset+size]
            self.inputs[name] = self.device[offset:offset+size]
            offset += size
        self.residual = cp.empty((m,w), cp.float32)
        self.x, self.branch, self.attention = [cp.empty((m,w), cp.float16) for _ in range(3)]
        self.qkv, self.layout = [cp.empty((m,3*w), cp.float16) for _ in range(2)]
        self.ff = cp.empty((m, a.ff_width), cp.float16)
        self.gate = cp.empty((m,a.intermediate), cp.float16)
        a.allocate_head(self)
        self.start, self.end = cp.cuda.Event(), cp.cuda.Event()

    def fill(self, items):
        f, k = self.fields, self.k
        self.host.fill(0)
        f["ids"].fill(self.a.pad)
        f["rows"].fill(-1)
        offset = 0
        for row, (ids, markers, kind) in enumerate(items):
            n = len(ids); end = offset+span(n)
            f["ids"][offset:offset+n] = ids
            f["positions"][offset:end] = self.runtime.positions[:end-offset]
            f["rows"][offset:end], f["kinds"][offset:end] = row, kind
            f["starts"][row], f["lengths"][row], f["counts"][row] = offset, n, len(markers)
            indices = f["indices"][row*(k+1):(row+1)*(k+1)]
            indices.fill(offset)
            indices[1:len(markers)+1] += markers
            offset = end
        self.device.set(self.host, stream=self.runtime.stream)

    def norm(self, target, prefix=None, branch=None, source=None, eps=None):
        source = self.residual if source is None else source
        weights = self.runtime.weights
        self.op.call("norm", source.shape[0], source, branch, weights.get(prefix+".weight") if prefix else None,
                     weights.get(prefix+".bias") if prefix else None, target, source.shape[1],
                     self.a.eps if eps is None else eps, int(target.dtype == cp.float32))

    def linear(self, x, target, prefix, packed=False):
        suffix = "_" if packed else "."
        self.op.linear(x, self.runtime.weights[prefix+suffix+"weight"], target,
                       self.runtime.weights.get(prefix+suffix+"bias"),self.linear_scratch)

    def attend(self, theta=None, window=-1):
        m,h,x = self.m,self.a.heads,self.inputs
        cosine,sine = self.runtime.phases[theta] if theta else (None,None)
        self.op.elements("layout", self.qkv.size, self.qkv,self.layout,cosine,sine,x["positions"],m,h)
        # One warp per 16 packed queries and head; four warps per block.
        self.op.call("attention", (h*m//16+3)//4, self.layout, self.attention, x["rows"], x["starts"],
                     x["lengths"], m, h, window, threads=128)

    def encoder(self):
        a,op,w = self.a,self.op,self.runtime.weights
        m,d = self.m,a.width
        op.elements("embed",m*d,self.inputs["ids"],w["encoder.embeddings.tok_embeddings.weight"],self.residual,m,d)
        self.norm(self.residual,"encoder.embeddings.norm")
        self.norm(self.x)
        for i in range(a.layers):
            p = f"encoder.layers.{i}."
            self.linear(self.x,self.qkv,p+"attn.Wqkv")
            local = i%a.global_every != 0
            self.attend(a.theta[int(local)],a.window if local else -1)
            self.linear(self.attention,self.branch,p+"attn.Wo")
            self.norm(self.x,p+"mlp_norm",self.branch)
            ff = self.ff.ravel()[:m*2*a.intermediate].reshape(m,2*a.intermediate)
            self.linear(self.x,ff,p+"mlp.Wi")
            op.elements("activate",m*a.intermediate,ff,self.gate,m*a.intermediate,a.intermediate,2)
            self.linear(self.gate,self.branch,p+"mlp.Wo")
            final = i==a.layers-1
            self.norm(self.residual if final else self.x,
                      "encoder.final_norm" if final else f"encoder.layers.{i+1}.attn_norm",self.branch)

    def forward(self):
        self.encoder()
        self.a.forward_head(self)

    def run(self, items, capture=True):
        self.fill(items)
        # Capturing costs about as much as one eager pass, so a new shape captures at once:
        # one slow request per shape class instead of an eager pass followed by a capture.
        if self.graph is None and capture:
            self.runtime.stream.begin_capture()
            try:
                self.forward()
            finally:
                self.graph = self.runtime.stream.end_capture()
        self.start.record(self.runtime.stream)
        if capture and self.graph is not None:
            self.graph.launch(self.runtime.stream)
        else:
            self.forward()
        self.end.record(self.runtime.stream)
        self.end.synchronize()
        logits,acts = self.a.read_outputs(self)
        return logits,acts,cp.cuda.get_elapsed_time(self.start,self.end)


class Runtime:
    def __init__(self, adapter, device, max_cached_shapes=16):
        self.adapter,self.device = adapter,cp.cuda.Device(device)
        self.slots,self.weights = OrderedDict(),{}
        self.limit,self.ops = max_cached_shapes,None
        self.pool = cp.cuda.MemoryPool()
        with self.device,cp.cuda.using_allocator(self.pool.malloc):
            self.stream = cp.cuda.Stream(non_blocking=True)
            try:
                with self.stream:
                    self.ops = Ops(self.stream)
                    with safe_open(str(adapter.path/"model.safetensors"),framework="numpy") as f:
                        shapes = adapter.weight_shapes()
                        if set(f.keys()) != set(shapes):
                            raise ValueError("Checkpoint tensor names do not match the Laya adapter")
                        for name,shape in shapes.items():
                            if tuple(f.get_slice(name).get_shape()) != shape:
                                raise ValueError(f"Incompatible tensor shape: {name}")
                        for name in shapes:
                            if name=="temperature":
                                continue
                            fp32 = "norm" in name or name.startswith("scorer.0.") or name in ("type_emb.weight","encoder.embeddings.tok_embeddings.weight")
                            self.weights[name] = cp.asarray(f.get_tensor(name).astype(np.float32 if fp32 else np.float16))
                    # RoPE tables cover every position a packed row can hold.
                    self.positions = np.arange(span(adapter.max_len), dtype=np.int32)
                    self.phases = {}
                    for theta in adapter.theta:
                        freq = 1/(np.float32(theta)**(np.arange(0,64,2,dtype=np.float32)/64))
                        angle = self.positions[:,None].astype(np.float32)*freq
                        self.phases[theta] = cp.asarray(np.cos(angle)), cp.asarray(np.sin(angle))
                self.stream.synchronize()
            except BaseException:
                self.close()
                raise

    def predict(self, items, capture=True):
        key = classes(items)
        with self.device,cp.cuda.using_allocator(self.pool.malloc),self.stream:
            miss = key not in self.slots
            if miss:
                if len(self.slots)>=self.limit:
                    self.stream.synchronize()
                    self.slots.popitem(last=False)
                    self.pool.free_all_blocks()
                self.slots[key] = Slot(self,*key)
            self.slots.move_to_end(key)
            slot = self.slots[key]
            graph_capture = capture and slot.graph is None
            logits,acts,gpu = slot.run(items,capture)
        return logits[:len(items)],acts[:len(items)],{"gpu_ms":gpu,"shape":key,"cache_miss":miss,"graph_capture":graph_capture,
                           "graph_replay":capture and slot.graph is not None,"owned_gpu_bytes":self.pool.total_bytes()}

    def close(self):
        with self.device:
            self.stream.synchronize()
            self.slots.clear()
            self.weights.clear()
            self.phases = {}
            if self.ops:
                self.ops.close()
                self.ops = None
            self.pool.free_all_blocks()
