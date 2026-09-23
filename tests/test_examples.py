import json
import math

import pytest

from benchmarks.backends import Jev, validate_answers
from benchmarks.compare import task_metrics, summarize
from benchmarks.prepare import classification_case
from examples.snake import Snake, greedy_move, replay_html


def test_snake_seed_and_growth():
    a, b = Snake(7, 6), Snake(7, 6)
    assert a.state() == b.state()
    a.body, a.direction = [(3, 3), (2, 3), (1, 3)], "right"
    a.food = (4, 3)
    a.step("right")
    assert a.body[0] == (4, 3) and len(a.body) == 4 and a.score == 1
    a.step("right")
    a.step("right")
    assert a.done and a.reason == "collision"


def test_snake_tail_vacates_and_invalid_actions():
    game = Snake()
    game.body = [(1, 1), (1, 2), (2, 2), (2, 1)]
    game.direction, game.food = "right", (5, 5)
    game.step("right")
    assert not game.done and game.body[0] == (2, 1)
    with pytest.raises(ValueError):
        game.step("reverse")
    game.food = (3, 1)
    assert greedy_move(game) == "right"


def test_replay_escapes_script_content(tmp_path):
    target = tmp_path/"replay.html"
    replay_html([{"backend": "</script><script>alert(1)</script>", "seed": 0, "score": 0, "frames": []}], target)
    assert "</script><script>alert" not in target.read_text(encoding="utf-8")


def test_upstream_question_labels():
    case = classification_case("ag_news", {"text": "news", "label": 3}, [])
    assert case["gold"]["topic"]["label"] == "sci_tech"
    assert list(case["questions"]["topic"]["criteria"]) == ["world", "sports", "business", "sci_tech"]
    case = classification_case("banking77", {"text": "card", "label_text": "card_arrival"}, ["card_arrival", "cash_withdrawal"])
    assert case["gold"]["intent"]["label"] == "card arrival"


def test_jev_uses_decisions_and_keeps_usage(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []
    payload = {"model": "typesafe/jev-1.13-20260917", "answers": {"q": {"type": "noul", "noul": .8}},
               "usage": {"input_tokens": 12, "output_tokens": 0, "cost": .01}}
    class Connection:
        def __init__(self, host, timeout):
            assert host == "openrouter.ai"
        def request(self, method, path, body, headers):
            calls.append((method, path, json.loads(body)))
        def getresponse(self):
            return type("Response", (), {"status": 200, "read": lambda self: json.dumps(payload).encode()})()
        def close(self):
            pass
    monkeypatch.setattr("benchmarks.backends.http.client.HTTPSConnection", Connection)
    with Jev(max_requests=1) as model:
        model.predict("hello", {"q": {"type": "noul", "instructions": "True?"}})
        assert model.cost == .01 and model.last_metrics["resolved_model"].endswith("20260917")
        with pytest.raises(RuntimeError, match="budget"):
            model.predict("hello", {})
    assert len(calls) == 1 and calls[0][:2] == ("POST", "/api/alpha/decisions")
    assert calls[0][2]["model"] == "typesafe/jev-1.13"


@pytest.mark.parametrize("value", [float("nan"), -1, 1.01, "yes"])
def test_invalid_jev_probability(value):
    with pytest.raises(ValueError):
        validate_answers({"answers": {"q": {"type": "noul", "noul": value}}}, {"q": {"type": "noul"}})


def test_missing_jev_key_is_local_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        Jev()


def test_metrics_do_not_drop_failures_or_zero_confidence():
    gold = {"q": {"label": "true"}}
    records = [{"gold": gold, "error": "timeout"},
               {"gold": gold, "answers": {"q": {"type": "noul", "noul": 1.0}}}]
    result = task_metrics(records)
    assert result["accuracy_including_failures"] == .5 and result["failed_requests"] == 1
    assert result["probabilistic_decisions"] == 1 and result["ece_15_bins"] == 0


def test_comparison_refuses_different_requests(tmp_path):
    for index in range(2):
        (tmp_path/f"backend-r{index}.json").write_text(json.dumps({"cases_sha256": "same", "backend": "cuda",
            "records": [{"id": "one", "input_sha256": str(index)}]}))
    with pytest.raises(ValueError, match="requests differ"):
        summarize(tmp_path)


def test_scalar_score_does_not_invent_a_class_prediction():
    result = task_metrics([{"gold": {"q": {"label": "1", "score": 1}},
                            "answers": {"q": {"type": "score", "score": 1.2}}}])
    assert result["score_mae"] == pytest.approx(.2)
    assert result["accuracy_including_failures"] is None
    assert result["unavailable_class_predictions"] == 1
