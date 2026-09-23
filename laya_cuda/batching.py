"""Bounded cross-request batching with one engine-owning worker."""
from concurrent.futures import Future
from copy import deepcopy
import math
from queue import Full
from threading import Condition, Thread, current_thread
import time
from typing import Any, cast

from .engine import Engine


class _Prediction(Future):
    metrics: dict


class BatchEngine:
    def __init__(self, model="laya", *, max_batch_size=16, max_batch_tokens=4096,
                 max_pending=128, max_wait_ms=1, **engine_options):
        for value in (max_batch_size,max_batch_tokens,max_pending):
            if type(value) is not int or value<1:
                raise ValueError("Batch and queue limits must be positive integers")
        if not math.isfinite(max_wait_ms) or max_wait_ms<0:
            raise ValueError("max_wait_ms must be finite and nonnegative")
        if engine_options.get("backend","cuda")!="cuda":
            raise ValueError("BatchEngine supports backend='cuda' only; use Engine for official")
        self.max_batch_size,self.max_batch_tokens = max_batch_size,max_batch_tokens
        self.max_pending,self.wait = max_pending,max_wait_ms/1000
        self._cv,self._queue,self._pending,self._closed = Condition(),[],0,False
        ready = Future()
        self._thread = Thread(target=self._worker,args=(model,engine_options,ready),name="laya-batch",daemon=True)
        self._thread.start()
        ready.result()

    def submit(self,state,questions):
        """Return a Future with per-request metrics; raise queue.Full at capacity."""
        started = time.perf_counter()
        with self._cv:
            if self._closed:
                raise RuntimeError("BatchEngine is closed")
            if self._pending>=self.max_pending:
                raise Full("BatchEngine pending request limit reached")
            future = _Prediction()
            future.metrics = {}
            self._queue.append((future,deepcopy(state),deepcopy(questions),started))
            self._pending += 1
            self._cv.notify()
        return future

    def close(self):
        """Reject new submissions and drain accepted requests before releasing CUDA."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if current_thread() is not self._thread:
            self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()

    def _finish(self,request,result=None,error=None):
        future = request[0]
        with self._cv:
            self._pending -= 1
        if not future.cancelled():
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)

    def _worker(self,model,options,ready):
        engine,active = None,[]
        try:
            engine = Engine(model,**options)
            from .runtime import span  # the worker already owns CUDA; keep package import light
            ready.set_result(None)
            while True:
                with self._cv:
                    self._cv.wait_for(lambda:self._queue or self._closed)
                    if not self._queue:
                        break
                    deadline = self._queue[0][3]+self.wait
                    while not self._closed and len(self._queue)<self.max_batch_size:
                        remaining = deadline-time.perf_counter()
                        if remaining<=0:
                            break
                        self._cv.wait(remaining)
                    active,self._queue = self._queue,[]
                prepared = []
                for request in active[:]:
                    if not request[0].set_running_or_notify_cancel():
                        self._finish(request)
                        active.remove(request)
                        continue
                    try:
                        items = engine.adapter.prepare(request[1],request[2])
                        # Questions are packed back to back, so any lengths can share a batch.
                        tokens = sum(span(len(x[0])) for x in items)
                        if len(items)>self.max_batch_size or tokens>self.max_batch_tokens:
                            raise ValueError("Request exceeds batch budget; increase limits or submit fewer questions")
                        prepared.append((request,items,tokens))
                    except Exception as error:
                        active.remove(request)
                        self._finish(request,error=error)
                while prepared:
                    batch,rest,count,tokens = [],[],0,0
                    for entry in prepared:
                        size = len(entry[1])
                        if count+size<=self.max_batch_size and tokens+entry[2]<=self.max_batch_tokens:
                            batch.append(entry)
                            count += size
                            tokens += entry[2]
                        else:
                            rest.append(entry)
                    prepared = rest
                    items = [item for _,rows,_ in batch for item in rows]
                    executed = time.perf_counter()
                    try:
                        logits,acts,metrics = cast(Any,engine.runtime).predict(items)
                    except Exception as error:
                        for request,_,_ in batch:
                            active.remove(request)
                            self._finish(request,error=error)
                        continue
                    offset = 0
                    for request,rows,_ in batch:
                        future,_,questions,started = request
                        try:
                            end = offset+len(rows)
                            result = engine.adapter.decode(logits[offset:end],acts[offset:end],questions,rows)
                            future.metrics = {**metrics,"batch_questions":count,"batch_requests":len(batch),
                                "queue_ms":(executed-started)*1000,"api_ms":(time.perf_counter()-started)*1000,
                                "tokens_per_question":[len(x[0]) for x in rows],"backend":"cuda","precision":"float16"}
                            active.remove(request)
                            self._finish(request,result)
                        except Exception as error:
                            if request in active:
                                active.remove(request)
                                self._finish(request,error=error)
                        offset += len(rows)
        except BaseException as error:
            with self._cv:
                self._closed = True
                active += self._queue
                self._queue = []
            if not ready.done():
                ready.set_exception(error)
            for request in active:
                if not request[0].running():
                    request[0].set_running_or_notify_cancel()
                self._finish(request,error=error)
        finally:
            if engine is not None:
                engine.close()
