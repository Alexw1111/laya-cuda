import json
import sys
from types import SimpleNamespace

import pytest

from laya_cuda.models import resolve


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "weights"
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"):
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    return path


def test_external_registry_paths_and_engine(checkpoint, tmp_path, monkeypatch):
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"custom": {"path": "weights"}}))
    monkeypatch.chdir(tmp_path.parent)
    assert resolve("custom", registry=config) == checkpoint
    assert resolve(checkpoint) == checkpoint
    import laya_cuda.engine as api
    runtime = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(api, "adapter", lambda path: SimpleNamespace(official=lambda device: runtime))
    with api.Engine("custom", registry=config, backend="official") as engine:
        assert engine.path == checkpoint


def test_registry_merge_override_and_revision(checkpoint, tmp_path, monkeypatch):
    calls = []
    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(checkpoint)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    assert resolve("laya") == checkpoint
    assert calls[-1][0] == "convaiinnovations/laya"
    assert calls[-1][1]["revision"] == "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"laya": {"repo_id": "example/custom", "revision": "pinned"}}))
    resolve("laya", registry=config)
    assert calls[-1][0] == "example/custom" and calls[-1][1]["revision"] == "pinned"
    resolve("laya", revision="explicit", registry=config)
    assert calls[-1][1]["revision"] == "explicit"
    resolve("laya-multilingual", registry=config)
    assert calls[-1][0] == "convaiinnovations/laya-multilingual"
    config.write_text(json.dumps({"laya": {"path": "weights"}}))
    assert resolve("laya", registry=config) == checkpoint
    assert len(calls) == 4  # A local override must not download anything.


@pytest.mark.parametrize("data", [[], {"bad": {}}, {"bad": {"repo_id": "repo"}},
    {"bad": {"path": " "}}, {"bad": {"path": 1}}, {"": {"path": "weights"}},
    {"bad": {"path": "weights", "repo_id": "repo", "revision": "rev"}}])
def test_invalid_registry(tmp_path, data):
    config = tmp_path / "models.json"
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="registry|entry"):
        resolve("bad", registry=config)


def test_registry_errors(tmp_path):
    config = tmp_path / "models.json"
    with pytest.raises(FileNotFoundError):
        resolve("custom", registry=config)
    config.write_text("not JSON")
    with pytest.raises(json.JSONDecodeError):
        resolve("custom", registry=config)
    config.write_text(json.dumps({"custom": {"path": "missing"}}))
    with pytest.raises(FileNotFoundError, match="Incomplete checkpoint"):
        resolve("custom", registry=config)
    with pytest.raises(FileNotFoundError, match="Unknown alias"):
        resolve("unknown", registry=config)
