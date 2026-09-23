from pathlib import Path

import numpy as np
import pytest

from laya_cuda import Engine

pytestmark=pytest.mark.gpu
Q={"q":{"type":"choice","instructions":"Choose the topic","criteria":["payment","software"]}}


@pytest.mark.parametrize("model",["laya","laya-multilingual","laya-typed-decisions"])
def test_eager_graph_lifecycle(model):
    path=Path("models")/model
    if not path.exists():
        pytest.skip("local checkpoint required")
    with Engine(path,max_cached_shapes=1) as e:
        for state in ("refund", "software error "*150, "支付退款"):
            items=e.adapter.prepare(state,Q)
            raw=e.runtime.predict(items,capture=False)
            captured=e.runtime.predict(items,capture=True)
            np.testing.assert_array_equal(raw[0],captured[0])
            np.testing.assert_array_equal(raw[1],captured[1])
            results=[e.predict(state,Q) for _ in range(4)]
            assert all(r==results[0] for r in results)
            assert len(e.runtime.slots)==1
        # Reusing one graph must upload new contents, not reuse the previous answer.
        for state in ("refund payment", "software error", "refund payment"):
            items=e.adapter.prepare(state,Q)
            captured=e.runtime.predict(items,capture=True)
            eager=e.runtime.predict(items,capture=False)
            np.testing.assert_array_equal(captured[0],eager[0])
            np.testing.assert_array_equal(captured[1],eager[1])
    assert e.runtime.pool.used_bytes()==0
    e.close()
    with pytest.raises(RuntimeError,match="closed"):
        e.predict("refund",Q)


def test_cold_shape_captures_on_first_use_and_eviction():
    path = Path("models/laya")
    if not path.exists():
        pytest.skip("local checkpoint required")
    with Engine(path, max_cached_shapes=1) as engine:
        first = engine.predict("refund", Q)
        assert engine.last_metrics["cache_miss"] and engine.last_metrics["graph_capture"]
        assert engine.last_metrics["graph_replay"]
        assert engine.predict("refund", Q) == first
        assert not engine.last_metrics["cache_miss"] and not engine.last_metrics["graph_capture"]
        assert engine.last_metrics["graph_replay"]
        engine.predict("software error "*150, Q)
        assert engine.predict("refund", Q) == first
        assert engine.last_metrics["cache_miss"] and engine.last_metrics["graph_capture"]
