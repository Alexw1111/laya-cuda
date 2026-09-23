import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.compare import summarize
from benchmarks.reproduce import checkpoint_files, digest, freeze, verify


def write_run(path, *, failed=False):
    rows = [{"id": "case", "input_sha256": "fixed", "suite": "tiny", "gold": {"q": {"label": "true"}},
             "ms": 2., **({"error": "timeout"} if failed else {"answers": {"q": {"type": "noul", "noul": .9}}})}]
    path.write_text(json.dumps({"backend": "cuda", "round": 0, "cases_sha256": "same", "records": rows}), encoding="utf-8")


def test_report_handles_missing_metrics_and_failures(tmp_path):
    pytest.importorskip("matplotlib")  # charts need the benchmark extra
    from benchmarks.report import render
    write_run(tmp_path/"cuda-r0.json", failed=True)
    report = render(tmp_path, '<script>alert("test")</script>')
    content = report.read_text(encoding="utf-8")
    assert report.name == "report.md" and "![Prediction latency](latency.png)" in content
    assert '<script>alert("test")</script>' not in content and "&lt;script&gt;" in content
    assert "<html" not in content and "<table" not in content
    assert "N/A" in content and "provenance incomplete" in content
    assert (tmp_path/"latency.svg").is_file() and (tmp_path/"metrics.csv").is_file()
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert summary["backends"]["cuda"]["tiny"][0]["failed_requests"] == 1


def test_expected_run_matrix_rejects_missing_worker(tmp_path):
    write_run(tmp_path/"cuda-r0.json")
    (tmp_path/"experiment.json").write_text(json.dumps({"backends": ["cuda", "upstream"], "rounds": 1, "requests_per_worker": 1}))
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(tmp_path)


def test_changed_result_refused(tmp_path):
    path = tmp_path/"cuda-r0.json"
    write_run(path)
    (tmp_path/"experiment.json").write_text(json.dumps({"backends": ["cuda"], "rounds": 1, "requests_per_worker": 1,
                                                        "result_hashes": {path.name: digest(path)}}))
    write_run(path, failed=True)
    with pytest.raises(ValueError, match="Recorded result changed"):
        summarize(tmp_path)


def test_freeze_and_tamper_detection(tmp_path, monkeypatch):
    import benchmarks.reproduce as reproduction
    root = tmp_path/"source"
    root.mkdir()
    for name in ("laya_cuda", "benchmarks"):
        (root/name).mkdir()
        (root/name/"__init__.py").write_text("# frozen source\n")
    for name in ("pyproject.toml", "uv.lock", ".python-version", "README.md", "LICENSE"):
        (root/name).write_text("test")
    inputs = tmp_path/"cases.jsonl"
    inputs.write_text('{"id":"one"}\n')
    monkeypatch.setattr(reproduction, "ROOT", root)
    monkeypatch.setattr(reproduction, "environment", lambda: {})
    args = SimpleNamespace(output=tmp_path/"out", cases_file=inputs, checkpoint_source=None, checkpoint="laya",
                           backends=["jev"], rounds=3, warmup=5, limit=None, jev_model="typesafe/jev-1.13", max_cost=1.)
    bundle = freeze(args)
    manifest = verify(bundle)
    assert manifest["protocol"]["rounds"] == 3
    from benchmarks.reproduce import pack
    import zipfile
    (bundle/"project/.env").write_text("private-key")
    with zipfile.ZipFile(pack(args.output)) as archive:
        assert "reproduce/inputs.jsonl" in archive.namelist()
        assert not any(name.endswith(".env") for name in archive.namelist())
    (bundle/"inputs.jsonl").write_text("modified")
    with pytest.raises(ValueError, match="mismatch"):
        verify(bundle)


def test_checkpoint_fingerprint_includes_tokenizer(tmp_path):
    (tmp_path/"tokenizer").mkdir()
    tokenizer = tmp_path/"tokenizer/tokenizer.json"
    tokenizer.write_text("first")
    before = checkpoint_files(tmp_path)
    tokenizer.write_text("second")
    assert before != checkpoint_files(tmp_path)


def test_manifest_path_cannot_escape_bundle(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps({"schema": 1, "files": {"../private": "bad"}}))
    with pytest.raises(ValueError, match="mismatch"):
        verify(tmp_path)


def test_cache_diagnostics_does_not_label_unknown_as_hit():
    from benchmarks.report import cache_diagnostics
    rows = [{"suite": "test", "ms": 2., "metrics": {"cache_miss": False, "gpu_ms": 1.}},
            {"suite": "test", "ms": 50., "metrics": {"cache_miss": True, "gpu_ms": 3.}},
            {"suite": "test", "ms": 4., "metrics": {}}]
    result = cache_diagnostics([{"backend": "cuda", "round": 0, "records": rows}])[0]
    assert (result["hits"], result["misses"], result["unknown_cache"]) == (1, 1, 1)
    assert result["hit_p99_ms"] == 2. and result["miss_p99_ms"] == 50.
    assert result["slowest_api_ms"] == 50. and result["gpu_at_slowest_ms"] == 3.
