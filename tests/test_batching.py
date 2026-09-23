from concurrent.futures import ThreadPoolExecutor, wait
from queue import Full
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from laya_cuda import BatchEngine
import laya_cuda.batching as batching


@pytest.fixture
def fake(monkeypatch):
    calls,owners = [],[]
    entered,release = Event(),Event()
    release.set()
    class Engine:
        def __init__(self,*args,**kwargs):
            owners.append(get_ident())
            self.adapter = self
            self.runtime = self
        def prepare(self,state,questions):
            owners.append(get_ident())
            if not questions:
                raise ValueError("empty questions")
            return [([state]*definition,[0],0) for definition in questions.values()]
        def predict(self,items):
            owners.append(get_ident())
            entered.set()
            assert release.wait(10)
            calls.append([len(x[0]) for x in items])
            if items[0][0][0]=="fatal":
                raise KeyboardInterrupt("worker stopped")
            if items[0][0][0]=="fail":
                raise RuntimeError("device failure")
            return [x[0][0] for x in items],[0]*len(items),{"gpu_ms":1}
        def decode(self,logits,acts,questions,items):
            return dict(zip(questions,logits))
        def close(self):
            owners.append(get_ident())
    monkeypatch.setattr(batching,"Engine",Engine)
    return SimpleNamespace(calls=calls,owners=owners,entered=entered,release=release)


def test_merge_restore_snapshot_and_owner(fake):
    with BatchEngine(max_wait_ms=30) as engine:
        questions={"same":20,"second":22}
        a=engine.submit("a",questions)
        questions.clear()
        b=engine.submit("b",{"same":21})
        assert a.result(5)=={"same":"a","second":"a"}
        assert b.result(5)=={"same":"b"}
        assert a.metrics["batch_requests"]==2
        assert a.metrics["api_ms"]>=a.metrics["queue_ms"]
    assert fake.calls==[[20,22,21]]
    assert len(set(fake.owners))==1 and fake.owners[0]!=get_ident()


def test_bounds_cancel_error_and_close(fake):
    fake.release.clear()
    engine=BatchEngine(max_pending=2,max_wait_ms=0)
    first=engine.submit("first",{"q":20})
    assert fake.entered.wait(5)
    second=engine.submit("cancel",{"q":20})
    assert second.cancel()
    with pytest.raises(Full):
        engine.submit("full",{"q":20})
    fake.release.set()
    engine.close()
    assert first.result()=={"q":"first"} and second.cancelled()
    assert engine._pending==0
    engine.close()
    with pytest.raises(RuntimeError,match="closed"):
        engine.submit("closed",{"q":20})


def test_packed_budget_and_request_failure(fake):
    with BatchEngine(max_batch_size=4,max_batch_tokens=512,max_wait_ms=30) as engine:
        futures=[engine.submit(str(i),{"q":length}) for i,length in enumerate([20,129,24,128])]
        assert [f.result(5) for f in futures]==[{"q":str(i)} for i in range(4)]
        for questions in ({},dict.fromkeys(range(5),20),{"q":513}):
            with pytest.raises(ValueError):
                engine.submit("bad",questions).result(5)
        with pytest.raises(RuntimeError,match="device failure"):
            engine.submit("fail",{"q":20}).result(5)
        assert engine.submit("ok",{"q":20}).result(5)=={"q":"ok"}
    # Packed rows need no common length: one batch within 4 questions and 512 tokens.
    assert fake.calls[0]==[20,129,24,128]
    assert engine._pending==0


def test_concurrent_producers_and_callback_close(fake):
    with BatchEngine(max_wait_ms=2) as engine:
        with ThreadPoolExecutor(8) as pool:
            futures=list(pool.map(lambda i:engine.submit(str(i),{"q":20}),range(64)))
        assert [f.result(5) for f in futures]==[{"q":str(i)} for i in range(64)]
        called=Event()
        def finish(future):
            engine.close()
            called.set()
        engine.submit("last",{"q":20}).add_done_callback(finish)
        assert called.wait(5)


def test_initialization_failure(fake,monkeypatch):
    def fail(*args,**kwargs):
        raise FileNotFoundError("missing checkpoint")
    monkeypatch.setattr(batching,"Engine",fail)
    with pytest.raises(FileNotFoundError):
        BatchEngine()
    for options in ({"max_wait_ms":float('nan')},{"max_pending":0},{"backend":"official"}):
        with pytest.raises(ValueError):
            BatchEngine(**options)


def test_fatal_worker_drains_and_notifies_cancelled_waiters(fake):
    fake.release.clear()
    engine=BatchEngine(max_wait_ms=0)
    failed=engine.submit('fatal',{'q':20})
    assert fake.entered.wait(5)
    cancelled=engine.submit('cancel',{'q':20})
    queued=engine.submit('queued',{'q':20})
    assert cancelled.cancel()
    fake.release.set()
    done,pending=wait([failed,cancelled,queued],timeout=5)
    assert not pending and len(done)==3
    for f in (failed,queued):
        with pytest.raises(KeyboardInterrupt):f.result()
    engine.close()
    assert engine._pending==0
