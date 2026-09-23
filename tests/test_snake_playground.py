import http.client
from http.server import ThreadingHTTPServer
import json
import random
import threading
import time

import pytest

from examples.snake import DIRECTIONS, Snake, cycle_cells, execute_decision, inspect_moves
from examples.snake_web import KEPT_FRAMES, Playground, handler_for


class Policy:
    last_metrics = {"gpu_ms": 1.}

    def __init__(self, action="left"):
        self.action = action
        self.calls = 0
        self.thread_ids = []
        self.closed = False

    def predict(self, state, questions):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        assert set(questions["move"]["criteria"]) == set(DIRECTIONS)
        planner = state.startswith("Safe route:")
        # Model only must not see the planner's verdicts.
        assert planner or not any(word in text for text in questions["move"]["criteria"].values() for word in ("Safe", "Unsafe", "Best"))
        return {"answers": {"move": {"choice": self.action,
                "probabilities": {k: .7 if k == self.action else .1 for k in DIRECTIONS}}}}

    def close(self):
        self.closed = True
        self.thread_ids.append(threading.get_ident())


def test_cycle_is_closed_and_covers_every_cell():
    for n in (6, 8, 12, 20):
        cells = cycle_cells(n)
        assert len(set(cells)) == n*n
        assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a, b in zip(cells, cells[1:]+cells[:1]))
    with pytest.raises(ValueError):
        Snake(size=7)


def test_shield_records_correction_and_pure_model_really_collides():
    shield, pure, policy = Snake(size=6), Snake(size=6), Policy("right")
    # This initial heading is left; right is the neck, so the shield must intervene.
    assert shield.direction == "left"
    frame = execute_decision(shield, policy)
    assert frame["proposed"] == "right" and frame["action"] != "right"
    assert frame["intervention"] and frame["shield_reason"] == "body"
    assert not shield.done and frame["after"]["steps"] == 1
    assert frame["state"]["steps"] == 0 and frame["step_ms"] >= frame["ms"]
    result = execute_decision(pure, policy, "model")
    assert pure.done and pure.reason == "collision" and not result["intervention"]
    assert policy.calls == 2


def test_cycle_guard_preserves_survival_with_adversarial_policy():
    for seed in range(6):
        rng, game, policy = random.Random(seed), Snake(seed, 6), Policy()
        for _ in range(1500):
            policy.action = rng.choice(list(DIRECTIONS))
            frame = execute_decision(game, policy)
            assert not frame["candidates"][frame["action"]]["blocked"]
            assert len(set(game.body)) == len(game.body)
            if game.done:
                break
        assert game.reason == "filled_board"
        assert game.score == 33


def test_tail_is_only_free_when_it_vacates():
    game = Snake(size=6)
    game.body = [(1, 1), (1, 2), (2, 2), (2, 1)]
    game.food = (5, 5)
    assert not game.collision((2, 1))
    assert not inspect_moves(game)["right"]["blocked"]
    game.food = (2, 1)
    assert game.collision((2, 1))


def test_playground_owns_model_on_one_thread_and_resets_mode():
    policy = Policy("right")
    owner = []
    def factory():
        owner.append(threading.get_ident())
        return policy
    app = Playground(factory=factory, size=6)
    try:
        result = app.command({"action": "step"})
        assert result["game"]["steps"] == 1 and result["interventions"] == 1
        first = json.dumps(app.command({"action": "export"}))
        assert '"proposed": "right"' in first
        app.command({"action": "mode", "mode": "model"})
        result = app.command({"action": "step"})
        assert result["game"]["reason"] == "collision" and result["interventions"] == 0
        app.command({"action": "play"})
        assert app.snapshot()["status"] == "finished"
        app.command({"action": "reset", "mode": "shield", "seed": 9})
        app.command({"action": "play"})
        time.sleep(.04)
        result = app.command({"action": "pause"})
        count = result["samples"]
        time.sleep(.04)
        assert app.snapshot()["samples"] == count
        with pytest.raises(ValueError):
            app.command({"action": "speed", "speed": -1})
        assert app.snapshot()["speed"] == 0  # unlimited by default
        for speed in (241, 2.5, True):
            with pytest.raises(ValueError):
                app.command({"action": "speed", "speed": speed})
        assert app.command({"action": "speed", "speed": 37})["speed"] == 37
    finally:
        app.close()
    assert policy.closed and not app.thread.is_alive()
    assert set(policy.thread_ids) == set(owner)


def test_playground_step_limit_lets_a_guarded_run_finish():
    app = Playground(factory=Policy, size=6)
    try:
        app.command({"action": "pause"})
        assert app.step_limit() == 6**4
    finally:
        app.close()


def test_commands_while_playing_do_not_bypass_pacing():
    app = Playground(factory=Policy, size=12)
    try:
        app.command({"action": "speed", "speed": 2})
        app.command({"action": "play"})
        started = app.snapshot()["game"]["steps"]
        finish = time.perf_counter() + .5
        while time.perf_counter() < finish:
            app.command({"action": "speed", "speed": 2})
            app.command({"action": "play"})
        # At 2 moves/s a half-second window allows at most two moves, however many commands arrive.
        assert app.snapshot()["game"]["steps"] - started <= 2
    finally:
        app.close()


@pytest.mark.parametrize("name", ["127.0.0.1", "localhost"])
def test_controls_accept_both_loopback_hosts(name):
    app = Playground(factory=Policy, size=6)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        app.command({"action": "pause"})
        host = f"{name}:{server.server_port}"
        def post(headers):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            connection.request("POST", "/api/control", json.dumps({"action": "pause"}),
                               {"Content-Type": "application/json", **headers})
            return connection.getresponse().status
        assert post({"Host": host, "Origin": f"http://{host}"}) == 200
        assert post({"Host": host, "Origin": "http://evil.example"}) == 403
        assert post({"Host": f"evil.example:{server.server_port}"}) == 403
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_unlimited_speed_is_not_paced():
    app = Playground(factory=Policy, size=12)
    try:
        app.command({"action": "speed", "speed": 0})
        app.command({"action": "play"})
        time.sleep(.3)
        # 60 moves/s would allow about 18 moves here; unlimited must go well beyond it.
        assert app.command({"action": "pause"})["game"]["steps"] > 60
    finally:
        app.close()


def test_board_size_is_selectable_and_frames_stay_bounded():
    app = Playground(factory=Policy, size=6)
    try:
        result = app.command({"action": "reset", "seed": 3, "size": 24})
        assert result["game"]["board"] == [24, 24] and app.step_limit() == 24**4
        for size in (7, 26, "12"):
            with pytest.raises(ValueError):
                app.command({"action": "reset", "size": size})
        app.command({"action": "mode", "mode": "model"})
        assert app.snapshot()["game"]["board"] == [24, 24]
        app.command({"action": "mode", "mode": "shield"})
        for _ in range(2*KEPT_FRAMES+5):
            app.command({"action": "step"})
        exported = app.command({"action": "export"})[0]
        assert len(exported["frames"]) <= 2*KEPT_FRAMES
        assert exported["omitted_earlier_frames"] + len(exported["frames"]) == exported["steps"]
        assert app.snapshot()["samples"] == exported["steps"]
    finally:
        app.close()
