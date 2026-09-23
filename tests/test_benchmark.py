import json
import pytest

from benchmarks.run import MODES, REFERENCE, accuracy_test, summarize


@pytest.mark.parametrize("model", ["laya", "custom-finetune"])
def test_incomplete_sampling_never_promotes_speedup(tmp_path, model):
    row={"id":"case", "gold":{"q":{"label":"yes"}},"answers":{"q":{
        "type":"choice","choice":"yes","probabilities":{"yes":.9,"no":.1},"action":{"act_probability":1}}}}
    for mode in MODES:
        for r in range(1 if mode==REFERENCE else 3):
            time=1 if mode=="cuda" else 2
            data={"model":model,"mode":mode,"round":r,"quality":[row] if r==0 else [],
                "protocol":{"warmup":20,"samples":1},"timing":{"short":{
                    "p50":time,"p95":time,"p99":time,"ms":[time],"answers":row["answers"]}}}
            if r==0:
                data["long"]=[{"id":"long-0","tokens":600,"answers":row["answers"]}]
            (tmp_path/f"{model}-{mode}-r{r}.json").write_text(json.dumps(data))
    assert not summarize(tmp_path)["models"][model]["timing"]["short"]["verified_speedup"]
    for path in tmp_path.glob("*-r*.json"):
        data=json.loads(path.read_text())
        data["protocol"]["samples"]=200
        data["timing"]["short"]["ms"]*=200
        path.write_text(json.dumps(data))
    assert summarize(tmp_path)["models"][model]["timing"]["short"]["verified_speedup"]
    # A long-input numerical failure blocks promotion even when latency and the fixed set pass.
    path=tmp_path/f"{model}-cuda-r0.json"
    data=json.loads(path.read_text())
    original=json.dumps(data["long"])
    data["long"][0]["answers"]["q"]["probabilities"]={"yes":.88,"no":.12}
    path.write_text(json.dumps(data))
    summary=summarize(tmp_path)["models"][model]
    assert not summary["long"]["passed"] and not summary["timing"]["short"]["verified_speedup"]
    data["long"]=json.loads(original)
    path.write_text(json.dumps(data))
    for path in tmp_path.glob(f"{model}-cuda-r*.json"):
        data=json.loads(path.read_text())
        data["timing"]["short"]["p50"]=2/1.05
        path.write_text(json.dumps(data))
    assert not summarize(tmp_path)["models"][model]["timing"]["short"]["verified_speedup"]


def test_custom_registry_reaches_all_workers(tmp_path, monkeypatch):
    import benchmarks.run as benchmark
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"my-finetune": {"path": "weights"}}))
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark", "--models", "my-finetune",
        "--registry", str(registry), "--rounds", "1", "--output", str(tmp_path / "results")])
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _: True)
    commands = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(benchmark, "summarize", lambda _: {"models": {}})
    benchmark.main()
    assert len(commands) == len(MODES)
    for command in commands:
        assert command[command.index("--checkpoint") + 1] == "my-finetune"
        assert command[command.index("--registry") + 1] == str(registry.resolve())


def test_accuracy_floor_is_the_fp32_reference_not_bf16(tmp_path):
    gold={"q":{"label":"yes"}}
    answer=lambda choice: {"q":{"type":"choice","choice":choice,"probabilities":{"yes":.6,"no":.4} if choice=="yes" else {"yes":.4,"no":.6},
                               "action":{"act_probability":1}}}
    # CUDA and FP32 agree (wrong on the pseudo-label); BF16 rounding happens to flip to the gold label.
    for mode,choice in (("cuda","no"),(REFERENCE,"no"),("official-fp16","no"),("official","yes")):
        data={"model":"m","mode":mode,"round":0,"quality":[{"id":"x","gold":gold,"answers":answer(choice)}],
              "long":[{"id":"long-0","tokens":600,"answers":answer(choice)}],"timing":{}}
        (tmp_path/f"m-{mode}-r0.json").write_text(json.dumps(data))
    quality=summarize(tmp_path)["models"]["m"]["quality"]
    assert quality["passed"] and quality["accuracy"]["official"] > quality["accuracy"]["cuda"]


def test_accuracy_gate_ignores_near_tie_luck_but_not_a_real_deficit():
    def rows(outcomes):
        answer=lambda choice: {"q":{"type":"choice","choice":choice,"probabilities":{"yes":.5,"no":.5},
                                    "action":{"act_probability":1}}}
        return [{"id":str(i),"gold":{"q":{"label":"yes"}},"answers":answer("yes" if right else "no")}
                for i,right in enumerate(outcomes)]
    # One decision lost to the reference is chance level; five lost and none won is not.
    assert accuracy_test(rows([0]),rows([1]))=={"wins":0,"losses":1,"p_value":.5}
    assert accuracy_test(rows([0,0,0,1]),rows([1,1,1,0]))["p_value"]==5/16
    assert accuracy_test(rows([0]*5),rows([1]*5))["p_value"]==1/32
    assert accuracy_test(rows([1,0]),rows([1,0]))["p_value"]==1.0
