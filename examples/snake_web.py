"""Loopback-only Snake playground; one worker owns the CUDA model and game."""
import argparse
from bisect import insort
from collections import deque
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time

from benchmarks.backends import open_backend
from .snake import Snake, execute_decision

KEPT_FRAMES = 500  # recent decisions retained for display/export; 24x24 frames carry long bodies


class Playground:
    def __init__(self, checkpoint="models/laya-multilingual", *, factory=None, size=12, max_steps=None):
        self.factory = factory or (lambda: open_backend("cuda", checkpoint))
        self.checkpoint, self.size, self.max_steps = checkpoint, size, max_steps
        self.commands = Queue(maxsize=32)
        self.lock = threading.Lock()
        self.published = {"status": "loading", "error": None}
        self.thread = threading.Thread(target=self._run, name="snake-model", daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.published

    def command(self, payload):
        if payload.get("action") not in ("play", "pause", "step", "reset", "speed", "mode", "export", "close"):
            raise ValueError("Unknown command")
        future = Future()
        self.commands.put_nowait((payload, future))
        return future.result(timeout=30)

    def step_limit(self):
        # With cycle safety each food takes at most size**2 moves, so size**4 lets a guarded run finish.
        return self.max_steps or self.game.size**4

    def _reset(self, seed, mode, size):
        if type(seed) is not int or not 0 <= seed <= 999999:
            raise ValueError("Seed must be an integer from 0 to 999999")
        if mode not in ("shield", "model"):
            raise ValueError("Mode must be shield or model")
        if type(size) is not int or size % 2 or not 6 <= size <= 24:
            raise ValueError("Board size must be an even integer from 6 to 24")
        self.game, self.mode = Snake(seed, size), mode
        self.frames, self.latencies, self.times = [], [], deque(maxlen=60)
        self.interventions, self.error, self.playing = 0, None, False

    def _publish(self):
        self.best = max(self.best, self.game.score)
        values = self.latencies  # kept sorted incrementally; long unlimited runs would otherwise re-sort every move
        def quantile(q):
            if not values:
                return None
            offset = (len(values)-1)*q
            low = int(offset)
            return values[low]+(values[min(low+1, len(values)-1)]-values[low])*(offset-low)
        result = {"status": "error" if self.error else "finished" if self.game.done else "running" if self.playing else "paused",
                  "error": self.error, "backend": "cuda", "checkpoint": self.checkpoint, "mode": self.mode,
                  "seed": self.game.seed, "game": self.game.state(), "frame": self.frames[-1] if self.frames else None,
                  "best": self.best, "interventions": self.interventions, "speed": self.speed,
                  "load_ms": self.load_ms, "p50_ms": quantile(.5), "p95_ms": quantile(.95),
                  "samples": len(values), "history_ms": [f["ms"] for f in self.frames[-60:]],
                  "moves_per_second": (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times) > 1 else 0}
        with self.lock:
            self.published = result

    def _step(self, model):
        if self.game.done or self.error:
            return
        try:
            frame = execute_decision(self.game, model, self.mode)
            self.frames.append(frame)
            if len(self.frames) > 2*KEPT_FRAMES:
                del self.frames[:-KEPT_FRAMES]
            insort(self.latencies, frame["ms"])
            self.times.append(time.perf_counter())
            self.interventions += int(frame["intervention"])
            if self.game.steps >= self.step_limit() and not self.game.done:
                self.game.done, self.game.reason = True, "step_limit"
            if self.game.done:
                self.playing = False
        except Exception as error:
            self.error, self.playing = f"{type(error).__name__}: {error}", False

    def _run(self):
        model = None
        try:
            started = time.perf_counter()
            model = self.factory()
            self.load_ms = (time.perf_counter()-started)*1000
            self.best, self.speed = 0, 0  # unlimited by default
            self._reset(42, "shield", self.size)
            self._publish()
            deadline = time.perf_counter()
            while True:
                try:
                    payload, future = self.commands.get(timeout=max(0, deadline-time.perf_counter()) if self.playing else None)
                except Empty:
                    started = time.perf_counter()
                    self._step(model)
                    self._publish()
                    deadline = max(time.perf_counter(), started+(1/self.speed if self.speed else 0))
                    continue
                try:
                    action = payload["action"]
                    if action == "close":
                        future.set_result({"closed": True})
                        break
                    if action in ("reset", "mode"):
                        self._reset(payload.get("seed", self.game.seed), payload.get("mode", self.mode),
                                    payload.get("size", self.game.size))
                    elif action == "speed":
                        speed = payload.get("speed")
                        if type(speed) is not int or not (speed == 0 or 1 <= speed <= 240):
                            raise ValueError("Pace must be 0 (unlimited) or 1-240 moves/s")
                        self.speed = speed
                        self.times.clear()
                    elif action == "play":
                        if not self.playing and not self.game.done and not self.error:
                            self.playing = True
                            self.times.clear()
                            deadline = time.perf_counter()
                    elif action == "pause":
                        self.playing = False
                    elif action == "step":
                        self.playing = False
                        self._step(model)
                    elif action == "export":
                        future.set_result([{"backend": "cuda", "checkpoint": self.checkpoint,
                                            "seed": self.game.seed, "mode": self.mode, "score": self.game.score,
                                            "steps": self.game.steps, "reason": self.game.reason or "paused",
                                            "interventions": self.interventions, "frames": list(self.frames),
                                            "omitted_earlier_frames": self.game.steps-len(self.frames)}])
                        continue
                    # Other commands keep the pending deadline, so they cannot trigger unpaced moves.
                    self._publish()
                    future.set_result(self.snapshot())
                except Exception as error:
                    future.set_exception(error)
        except Exception as error:
            with self.lock:
                self.published = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        finally:
            if model:
                model.close()

    def close(self):
        if self.thread.is_alive():
            self.command({"action": "close"})
            self.thread.join(timeout=30)


def handler_for(playground):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, content_type="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type+"; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                page = Path(__file__).with_name("snake_replay.html").read_text(encoding="utf-8")
                self.reply(200, page.replace("__EPISODES__", "null"), "text/html")
            elif self.path == "/api/state":
                self.reply(200, playground.snapshot())
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            host = self.headers.get("Host")
            if (host not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")
                    or self.headers.get("Origin") not in (None, f"http://{host}")):
                self.reply(403, {"error": "Only same-origin local controls are accepted"})
                return
            if self.path != "/api/control" or self.headers.get("Content-Type") != "application/json":
                self.reply(400, {"error": "Expected a JSON control command"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 4096:
                    raise ValueError("Invalid command size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object")
                if playground.snapshot()["status"] == "loading":
                    self.reply(503, {"error": "Model is loading"})
                    return
                if not playground.thread.is_alive():
                    self.reply(503, {"error": "Model worker is unavailable"})
                    return
                self.reply(200, playground.command(payload))
            except Exception as error:
                self.reply(400, {"error": str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/laya-multilingual")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    playground = Playground(args.checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(playground))
    print(f"Snake playground: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        playground.close()


if __name__ == "__main__":
    main()
