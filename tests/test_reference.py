"""Input contract tests use the real SDK without loading its GPU model."""
from pathlib import Path

import pytest

from laya_cuda.laya import Laya, question

pytestmark = pytest.mark.reference


def test_reference_rejects_cpu_forward_inputs():
    import torch
    from laya_cuda.official import Official
    reference = object.__new__(Official)
    reference.torch = torch
    with pytest.raises(RuntimeError, match="CPU comparison"):
        reference._before_forward(None, (torch.ones(1),))


@pytest.mark.parametrize("model", ["laya", "laya-multilingual", "laya-typed-decisions"])
def test_official_input_contract(model):
    pytest.importorskip("laya")
    from laya.common import build_sequence, collate_items, QTYPES
    from laya.agent import Agent
    from transformers import PreTrainedTokenizerFast
    path = Path("models")/model
    if not path.exists():
        pytest.skip("local checkpoint required")
    adapter = Laya(path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path/"tokenizer/tokenizer.json"),
        cls_token=adapter.tokenizer.id_to_token(adapter.cls), sep_token=adapter.tokenizer.id_to_token(adapter.sep),
        mask_token=adapter.mask_text, pad_token=adapter.tokenizer.id_to_token(adapter.pad))
    queries = {
        "a": {"type":"choice", "instructions":{"ask":"选择 [MASK]"}, "criteria":{"yes":False,"no":0}},
        "b": {"type":"score", "instructions":"Urgency?", "criteria":["low",{"level":2},"high"]},
        "c": {"type":"noul", "instructions":"支付退款?", "criteria":{"true":{"yes":True}}},
        "one": {"type":"choice", "instructions":"Choose", "criteria":["only"]},
        "many": {"type":"choice", "instructions":"Choose "*100, "criteria":{str(i):"long "*80 for i in range(12)}},
    }
    for state in ("", "支付😀退款 café [MASK]", {"turns":["hello",False]}, "boundary "*3000):
        actual = adapter.prepare(state,queries)
        expected = []
        for definition in queries.values():
            q = Agent._to_internal(definition)
            ids,markers = build_sequence(tokenizer,state,q,adapter.max_len,adapter.head_max_len)
            expected.append((ids,markers,QTYPES[q["t"]]))
        assert actual == expected
        packed = collate_items([[{"ids":ids,"markers":markers,"qtype":kind} for ids,markers,kind in expected]],adapter.pad)
        for row,(ids,markers,kind) in enumerate(actual):
            assert packed["input_ids"][row,:len(ids)].tolist() == ids
            assert packed["attention_mask"][row].sum().item() == len(ids)
            assert packed["marker_pos"][row,:len(markers)].tolist() == markers
            assert packed["marker_mask"][row].sum().item() == len(markers)
        assert len(question(queries["one"])[3]) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("model", ["laya", "laya-multilingual", "laya-typed-decisions"])
def test_official_unicode_and_option_outputs(model):
    pytest.importorskip("laya")
    import numpy as np
    from laya_cuda import Engine
    from benchmarks.run import probabilities
    path=Path("models")/model
    if not path.exists():
        pytest.skip("local checkpoint required")
    queries={
        "single":{"type":"choice","instructions":"Choose","criteria":["only"]},
        "many":{"type":"choice","instructions":"Select the best matching topic.","criteria":["payment","refund","software","shipping","security","weather","food","sports","music","science","other","unknown"]},
        "score":{"type":"score","instructions":"Urgency?","criteria":["low",{"level":"medium"},"high"]},
        "noul":{"type":"noul","instructions":"Is a refund requested?"},
    }
    with Engine(path) as candidate,Engine(path,backend="official") as reference:
        reference.runtime.agent.dtype=reference.runtime.torch.float16
        errors=[]
        for state in ("退款 😀 café — please refund my payment.", {"turns":["hello", "a software error", False]},"boundary "*3000):
            a,b=candidate.predict(state,queries),reference.predict(state,queries)
            assert a["usage"]==b["usage"]
            assert a["answers"]["single"]["probabilities"]=={"only":1.0}
            for q,x in a["answers"].items():
                y=b["answers"][q]
                assert x.keys()==y.keys()
                errors.extend(np.abs(np.array(probabilities(x))-probabilities(y)))
                errors.append(abs(x["action"]["act_probability"]-y["action"]["act_probability"]))
        assert np.isfinite(errors).all()
        assert np.mean(errors)<=1e-3 and max(errors)<=1e-2
