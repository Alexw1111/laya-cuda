"""JSON checkpoint configuration and a small model adapter registry."""
import json
from pathlib import Path

FILES = ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"]


def _checkpoints(registry=None):
    sources = [Path(__file__).with_name("models.json")]
    if registry is not None:
        sources.append(Path(registry).expanduser())
    entries = {}
    for source in sources:
        data = read_json(source)
        if not isinstance(data, dict):
            raise ValueError(f"Model registry must be a JSON object: {source}")
        for alias, entry in data.items():
            if (not alias.strip() or not isinstance(entry, dict)
                    or set(entry) not in ({"path"}, {"repo_id", "revision"})
                    or any(not isinstance(v, str) or not v.strip() for v in entry.values())):
                raise ValueError(f"Invalid model entry {alias!r} in {source}; use path or repo_id and revision")
            if "path" in entry:
                path = Path(entry["path"]).expanduser()
                entry = {"path": str((source.parent / path).resolve())}
            entries[alias] = entry
    return entries


def resolve(model, revision=None, *, registry=None):
    path = Path(model).expanduser()
    if not path.is_dir():
        checkpoints = _checkpoints(registry)
        if str(model) not in checkpoints:
            raise FileNotFoundError(f"Unknown alias or missing local checkpoint: {model}")
        entry = checkpoints[str(model)]
        if "path" in entry:
            path = Path(entry["path"])
        else:
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(entry["repo_id"], revision=revision or entry["revision"],
                                          allow_patterns=FILES))
    for name in FILES[:3] + ["tokenizer/tokenizer.json"]:
        if not (path / name).is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {path / name}")
    return path.resolve()


def adapter(path):
    from .laya import Laya
    registry = {"laya": Laya}
    kind = "laya" if (path / "rl_agent_config.json").is_file() else None
    if kind not in registry:
        raise ValueError("No compatible model adapter")
    return registry[kind](path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
