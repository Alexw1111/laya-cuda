import json
from pathlib import Path
import subprocess
import sys

import pytest

from laya_cuda import cli

QUESTIONS = {"urgent": {"type": "noul", "instructions": "Urgent?"}}


class StubEngine:
    opened = []

    def __init__(self, model, **options):
        self.model, self.options, self.closed = model, options, False
        self.last_metrics = {"gpu_ms": 1.0}
        StubEngine.opened.append(self)

    def predict(self, state, questions):
        if state == "boom":
            raise ValueError("bad state")
        return {"answers": {k: {"type": "noul", "noul": .25} for k in questions}, "state_echo": state}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


@pytest.fixture
def stub(monkeypatch):
    StubEngine.opened = []
    monkeypatch.setattr(cli, "Engine", StubEngine)
    return StubEngine


def test_version_help_and_usage_errors(capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main(["--version"])
    assert exit.value.code == 0 and f"laya-cuda {cli.__version__}" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit:
        cli.main(["predict", "--nope"])
    assert exit.value.code == 2
    with pytest.raises(SystemExit) as exit:
        cli.main([])
    assert exit.value.code == 2


def test_import_does_not_load_gpu_or_reference_frameworks():
    code = "import sys, laya_cuda.cli; print(sorted({'cupy', 'torch', 'transformers'} & set(sys.modules)))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"


def test_predict_single_state(stub, capsys):
    assert cli.main(["predict", "refund", "-m", "m", "-q", json.dumps(QUESTIONS), "--metrics"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["answers"]["urgent"]["noul"] == .25 and result["metrics"] == {"gpu_ms": 1.0}
    engine = stub.opened[0]
    assert engine.model == "m" and engine.options["backend"] == "cuda" and engine.closed


def test_predict_jsonl_records_failures_and_keeps_ids(stub, tmp_path, capsys):
    source = tmp_path/"in.jsonl"
    source.write_text('{"id": "a", "state": "x"}\n\nnot json\n{"id": "c", "state": "boom"}\n'
                      '{"state": "y", "questions": {"q": {"type": "noul", "instructions": "?"}}}\n', encoding="utf-8")
    questions = tmp_path/"q.json"
    questions.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    output = tmp_path/"out.jsonl"
    assert cli.main(["predict", "-i", str(source), "-q", str(questions), "-o", str(output)]) == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["a", 3, "c", 5]
    assert "error" in rows[1] and rows[2]["error"] == "ValueError: bad state" and "q" in rows[3]["answers"]
    assert len(stub.opened) == 1 and stub.opened[0].closed
    assert "2 request(s) failed" in capsys.readouterr().err


def test_predict_reports_bad_invocations_without_traceback(stub, capsys):
    assert cli.main(["predict", "x", "-q", "missing.json"]) == 1
    assert cli.main(["predict", "x"]) == 1
    assert cli.main(["predict", "-q", "{not json"]) == 1
    assert cli.main(["predict", "x", "-i", "in.jsonl", "-q", json.dumps(QUESTIONS)]) == 1
    err = capsys.readouterr().err
    assert "questions file not found" in err and "--questions is required" in err
    assert "not valid JSON" in err and "either a STATE or --input" in err
    assert "Traceback" not in err and not stub.opened


def test_models_lists_builtin_names(capsys):
    assert cli.main(["models"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines] == ["laya", "laya-multilingual", "laya-typed-decisions"]
    assert "convaiinnovations/laya@1c5edc17a7ac" in lines[0]


@pytest.mark.gpu
def test_doctor_on_the_local_gpu(capsys):
    model = Path("models/laya")
    code = cli.main(["doctor", "--json"] + ([str(model)] if model.is_dir() else []))
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["passed"], report
    assert {c["check"] for c in report["checks"]} >= {"CuPy", "driver", "GPU 0", "NVRTC", "cuBLAS", "kernels"}
