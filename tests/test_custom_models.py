"""Synthetic checkpoints exercise extension mechanics, not trained-model quality."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from safetensors.numpy import save_file

from laya_cuda import Engine, BatchEngine
from laya_cuda.laya import Laya


@pytest.mark.gpu
def test_custom_checkpoint_weights_registry_and_batching(tmp_path):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("GPU required")
    source = Path("models/laya")
    if not source.exists():
        pytest.skip("Local tokenizer and configuration required")
    custom = tmp_path / "custom-weights"
    custom.mkdir()
    shutil.copytree(source / "tokenizer", custom / "tokenizer")
    (custom / "encoder").mkdir()
    config = json.loads((source / "encoder/config.json").read_text())
    config.update(hidden_size=64, num_attention_heads=1, num_hidden_layers=1,
                  intermediate_size=128, layer_types=["full_attention"])
    (custom / "encoder/config.json").write_text(json.dumps(config))
    task = json.loads((source / "rl_agent_config.json").read_text())
    task.update(head_layers=1, max_len=128, head_max_len=64)
    (custom / "rl_agent_config.json").write_text(json.dumps(task))
    adapter = Laya(custom)
    weights = {name: np.zeros(shape, np.float32) for name, shape in adapter.weight_shapes().items()}
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"my-finetune": {"path": "custom-weights"}}))
    queries = {"q": {"type": "choice", "instructions": "Choose", "criteria": ["yes", "no"]}}
    outputs = []
    for bias in ([-4, 4], [4, -4]):
        weights["act_head.2.bias"] = np.array(bias, np.float32)
        save_file(weights, str(custom / "model.safetensors"))
        with Engine("my-finetune", registry=registry) as engine:
            output = engine.predict("custom input", queries)
            assert engine.path == custom
            np.testing.assert_array_equal(engine.runtime.weights["act_head.2.bias"].get(), bias)
            assert output == engine.predict("custom input", queries)
            outputs.append(output["answers"]["q"]["action"]["act_probability"])
        assert engine.runtime.pool.used_bytes() == 0
    assert outputs[0] < .001 and outputs[1] > .999
    with Engine(custom) as engine:
        expected = engine.predict("custom input", queries)
    with BatchEngine("my-finetune", registry=registry) as engine:
        assert engine.submit("custom input", queries).result(30) == expected
    weights["act_head.2.bias"] = np.zeros(3, np.float32)
    save_file(weights, str(custom / "model.safetensors"))
    with pytest.raises(ValueError, match="Incompatible tensor shape"):
        Engine("my-finetune", registry=registry)
