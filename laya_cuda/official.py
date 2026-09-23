"""Optional thin adapter around the installed, unmodified official SDK."""
import gc
import shutil
import tempfile
from pathlib import Path


class Official:
    def __init__(self,path,device):
        try:
            import torch
            import laya
        except ImportError as error:
            raise ImportError('Install "laya-cuda[reference]" to use the official backend') from error
        if not torch.cuda.is_available():
            raise RuntimeError("The official backend requires a CUDA-enabled Torch wheel; the installed Torch runtime has no CUDA support. See the Windows full-install limitation in README.md.")
        self.torch = torch
        # The SDK repairs tokenizer configuration in place. Give it a disposable
        # view so immutable checkpoints and user files are never rewritten.
        self.temp = tempfile.TemporaryDirectory(prefix="laya-cuda-reference-")
        target = Path(self.temp.name)
        try:
            shutil.copytree(path/"tokenizer",target/"tokenizer")
            shutil.copytree(path/"encoder",target/"encoder")
            shutil.copy2(path/"rl_agent_config.json",target/"rl_agent_config.json")
            try:
                (target/"model.safetensors").hardlink_to(path/"model.safetensors")
            except OSError:
                shutil.copy2(path/"model.safetensors",target/"model.safetensors")
            self.agent = laya.Agent(str(target),device=device)
            if self.agent.device.type != "cuda":
                raise RuntimeError("Official SDK unexpectedly fell back to CPU")
            parameters = list(self.agent.model.parameters()) + list(self.agent.model.buffers())
            if not parameters or any(p.device.type != "cuda" for p in parameters):
                raise RuntimeError("Official model tensors are not all on CUDA")
            self.device_info = {"torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                                "model_devices": sorted({str(p.device) for p in parameters}),
                                "gpu": torch.cuda.get_device_name(self.agent.device)}
            self.events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            self.hooks = [self.agent.model.register_forward_pre_hook(self._before_forward),
                          self.agent.model.register_forward_hook(lambda *_: self.events[1].record(torch.cuda.current_stream(device)))]
        except BaseException:
            self.temp.cleanup()
            raise

    @property
    def precision(self):
        return str(self.agent.dtype)

    def _before_forward(self, module, inputs):
        tensors = [value for value in inputs if self.torch.is_tensor(value)]
        if not tensors or any(value.device.type != "cuda" for value in tensors):
            raise RuntimeError("Official forward inputs are not on CUDA; refusing CPU comparison")
        self.device_info["input_devices"] = sorted({str(value.device) for value in tensors})
        self.events[0].record(self.torch.cuda.current_stream(self.agent.device))

    def predict(self,state,questions):
        result = self.agent.system_one(state,questions)
        if self.agent.device.type != "cuda":
            raise RuntimeError("Official SDK fell back to CPU; refusing misleading GPU results")
        self.torch.cuda.synchronize(self.agent.device)
        self.gpu_ms = self.events[0].elapsed_time(self.events[1])
        return result

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.agent = None
        self.events = None
        gc.collect()
        self.torch.cuda.empty_cache()
        self.temp.cleanup()
