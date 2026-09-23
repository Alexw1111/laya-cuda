"""Public synchronous inference API with explicit backend selection."""
import time

from .models import adapter, resolve


class Engine:
    def __init__(self, model="laya", *, revision=None, device="cuda:0", backend="cuda", max_cached_shapes=16,
                 registry=None):
        if backend not in ("cuda", "official"):
            raise ValueError("backend must be 'cuda' or 'official'")
        if type(max_cached_shapes) is not int or not 1<=max_cached_shapes<=64:
            raise ValueError("max_cached_shapes must be an integer between 1 and 64")
        if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("v1 requires a CUDA device")
        self.backend,self.device,self.closed = backend,device,False
        self.last_metrics = {}
        self.path = resolve(model,revision,registry=registry)
        self.adapter = adapter(self.path)
        if backend == "cuda":
            from .runtime import Runtime
            self.runtime = Runtime(self.adapter,int(device.split(":")[-1]) if ":" in device else 0,max_cached_shapes)
        else:
            self.runtime = self.adapter.official(device)

    def predict(self,state,questions):
        if self.closed:
            raise RuntimeError("Engine is closed")
        started = time.perf_counter()
        if self.backend == "official":
            result = self.runtime.predict(state,questions)
            metrics = {"precision":self.runtime.precision,"gpu_ms":self.runtime.gpu_ms}
        else:
            items = self.adapter.prepare(state,questions)
            logits,acts,metrics = self.runtime.predict(items)
            result = self.adapter.decode(logits,acts,questions,items)
            metrics.update(precision="float16",tokens_per_question=[len(v[0]) for v in items],
                           input_sha256=self.adapter.fingerprint(items))
        self.last_metrics = {**metrics,"backend":self.backend,"api_ms":(time.perf_counter()-started)*1000}
        return result

    def close(self):
        if not self.closed:
            self.runtime.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()
