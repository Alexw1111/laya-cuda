"""Comparison clients, kept outside the installed inference library."""
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

UPSTREAM_REVISION = "573e5b62696ba441230cd6be71d593331b5d23af"
JEV_MODEL = "typesafe/jev-1.13"


def validate_answers(result, questions):
    answers = result.get("answers", {})
    if set(answers) != set(questions):
        raise ValueError("Response question IDs do not match the request")
    for key, question in questions.items():
        answer = answers[key]
        kind = question["type"]
        if answer.get("type") != kind:
            raise ValueError(f"Wrong answer type: {key}")
        if kind == "noul":
            value = answer.get("noul")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid probability: {key}")
        else:
            labels = list(question["criteria"]) if kind == "choice" else list(map(str, range(len(question["criteria"]))))
            if kind == "choice" and answer.get("choice") not in labels:
                raise ValueError(f"Unknown choice: {key}")
            if kind == "score":
                value = answer.get("score")
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= len(labels)-1:
                    raise ValueError(f"Invalid score: {key}")
            probabilities = answer.get("probabilities")
            if probabilities is not None:
                if set(probabilities) != set(labels):
                    raise ValueError(f"Probability labels do not match: {key}")
                values = list(probabilities.values())
                if any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
                    raise ValueError(f"Invalid distribution: {key}")
                if abs(sum(values)-1) > .02:
                    raise ValueError(f"Distribution does not sum to one: {key}")
    return result


class Jev:
    """Synchronous Decisions API client with connection reuse and no hidden retries."""
    def __init__(self, model=JEV_MODEL, timeout=30, max_requests=100, max_cost=1.0):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY to run Jev; offline/local backends need no key")
        if max_requests < 1 or not math.isfinite(max_cost) or max_cost <= 0 or timeout <= 0:
            raise ValueError("Request, cost and timeout limits must be positive")
        self.model, self.max_requests, self.max_cost = model, max_requests, max_cost
        self.requests, self.cost, self.last_metrics = 0, 0.0, {}
        self.headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        self.connection = http.client.HTTPSConnection("openrouter.ai", timeout=timeout)

    def predict(self, state, questions):
        if self.requests >= self.max_requests or self.cost >= self.max_cost:
            raise RuntimeError("Jev request or reported-cost budget exhausted")
        payload = json.dumps({"model": self.model, "state": state, "questions": questions}, ensure_ascii=False).encode()
        self.requests += 1
        started = time.perf_counter()
        self.last_metrics = {}
        try:
            self.connection.request("POST", "/api/alpha/decisions", payload, self.headers)
            response = self.connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"OpenRouter Decisions returned HTTP {response.status}; request was not retried")
            result = json.loads(body)
            cost = result.get("usage", {}).get("cost")
            if cost is not None:
                if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
                    raise ValueError("Invalid reported cost")
                self.cost += cost
            self.last_metrics = {"api_ms": (time.perf_counter()-started)*1000,
                                 "usage": result.get("usage"), "resolved_model": result.get("model"),
                                 "provider": result.get("provider"), "request_id": result.get("id"),
                                 "reported_cost_total": self.cost}
            return validate_answers(result, questions)
        except Exception:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def upstream_source(directory):
    directory = Path(directory).resolve()
    revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION or subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain", "--", "laya"], text=True).strip():
        raise ValueError("Upstream checkout must match the recorded revision with unchanged laya sources")
    loaded = sys.modules.get("laya")
    if loaded and not Path(loaded.__file__).resolve().is_relative_to(directory):
        raise RuntimeError("Run upstream comparison in a fresh process; a different laya SDK is loaded")
    sys.path.insert(0, str(directory))
    return {"repository": "https://github.com/nandhakishorm/laya", "revision": revision,
            "files": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (directory/"laya").glob("*.py")}}


def open_backend(name, checkpoint="models/laya", *, upstream="vendor/laya-upstream", **jev_options):
    if name == "jev":
        return Jev(**jev_options)
    if name not in ("cuda", "official", "upstream"):
        raise ValueError(f"Unknown backend: {name}")
    provenance = upstream_source(upstream) if name == "upstream" else None
    from laya_cuda import Engine
    engine = Engine(checkpoint, backend="cuda" if name == "cuda" else "official")
    engine.comparison_provenance = provenance
    return engine
