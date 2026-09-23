"""Real model decisions, explicit cycle safety, and portable Snake recordings."""
import argparse
from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import time

from benchmarks.backends import open_backend

DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}  # option order measurably affects Laya choices
INSTRUCTIONS = "Choose the best safe move toward food."
RAW_INSTRUCTIONS = ("Choose an absolute direction for the next Snake move. Avoid blocked moves. Prefer a safe move "
                    "toward food with open space. UP decreases y; DOWN increases y; LEFT decreases x; RIGHT increases x.")


def cycle_cells(size):
    """A Hamiltonian cycle on an even square grid."""
    if size < 6 or size % 2:
        raise ValueError("Cycle safety requires an even board size of at least six")
    return ([(0, 0)] + [(x, y) for y in range(size)
                       for x in (range(1, size) if y % 2 == 0 else range(size-1, 0, -1))]
            + [(0, y) for y in range(size-1, 0, -1)])


@dataclass
class Snake:
    seed: int = 0
    size: int = 12
    body: list = field(init=False)
    direction: str = field(init=False)
    food: tuple | None = None
    score: int = 0
    steps: int = 0
    done: bool = False
    reason: str | None = None

    def __post_init__(self):
        self.cycle = cycle_cells(self.size)
        self.ranks = {cell: i for i, cell in enumerate(self.cycle)}
        middle = self.ranks[(self.size//2, self.size//2)]
        self.body = [self.cycle[(middle-i) % len(self.cycle)] for i in range(3)]
        delta = tuple(self.body[0][i]-self.body[1][i] for i in range(2))
        self.direction = next(k for k, v in DIRECTIONS.items() if v == delta)
        self.rng = random.Random(self.seed)
        self.spawn_food()

    def spawn_food(self):
        body = set(self.body)
        empty = [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in body]
        self.food = self.rng.choice(empty) if empty else None
        if not empty:
            self.done, self.reason = True, "filled_board"

    def next_cell(self, action):
        if action not in DIRECTIONS:
            raise ValueError("Expected an absolute direction: up, right, down or left")
        dx, dy = DIRECTIONS[action]
        return action, (self.body[0][0]+dx, self.body[0][1]+dy)

    def collision(self, cell):
        occupied = self.body if cell == self.food else self.body[:-1]
        return not (0 <= cell[0] < self.size and 0 <= cell[1] < self.size) or cell in occupied

    def state(self):
        return {"board": [self.size, self.size], "snake_head_first": [list(v) for v in self.body],
                "heading": self.direction, "food": list(self.food) if self.food is not None else None,
                "score": self.score, "steps": self.steps, "done": self.done, "reason": self.reason}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is finished")
        direction, cell = self.next_cell(action)
        self.steps += 1
        self.direction = direction
        if self.collision(cell):
            self.done, self.reason = True, "collision"
            return
        grow = cell == self.food
        self.body.insert(0, cell)
        if grow:
            self.score += 1
            self.spawn_food()
        else:
            self.body.pop()


def inspect_moves(game):
    """Geometry facts and conservative cycle-order checks, not model estimates."""
    total = game.size**2
    head, tail = game.ranks[game.body[0]], game.ranks[game.body[-1]]
    gap = (tail-head) % total
    food_gap = (game.ranks[game.food]-head) % total
    moves = {}
    for action in DIRECTIONS:
        _, cell = game.next_cell(action)
        wall = not (0 <= cell[0] < game.size and 0 <= cell[1] < game.size)
        blocked, grow = game.collision(cell), cell == game.food
        reachable = set()
        if not blocked:
            occupied = set(game.body if grow else game.body[:-1])
            pending = deque([cell])
            reachable.add(cell)
            while pending:
                x, y = pending.popleft()
                for dx, dy in DIRECTIONS.values():
                    other = (x+dx, y+dy)
                    if (0 <= other[0] < game.size and 0 <= other[1] < game.size
                            and other not in occupied and other not in reachable):
                        reachable.add(other)
                        pending.append(other)
        jump = (game.ranks[cell]-head) % total if not wall else 0
        ordered = 0 < jump < gap or (jump == gap and not grow)
        safe = not blocked and ordered and jump <= food_gap
        reason = "wall" if wall else "body" if blocked else "cycle order" if not ordered else "would pass food" if jump > food_gap else "safe"
        moves[action] = {"cell": list(cell), "blocked": blocked, "shield_allowed": safe,
                         "reason": reason, "progress": jump if safe else None, "eats": grow, "reachable_cells": len(reachable),
                         "food_reachable": game.food in reachable,
                         "food_distance": abs(cell[0]-game.food[0])+abs(cell[1]-game.food[1])}
    return moves


def model_input(moves):
    """Planner verdicts go into each option's text, where Laya scores that option."""
    allowed = [a for a in moves if moves[a]["shield_allowed"]]
    best = max(allowed, key=lambda a: (moves[a]["progress"], -moves[a]["food_distance"]), default=None)
    reachable = any(m["food_reachable"] for m in moves.values() if not m["blocked"])
    state = f"Safe route: {'yes' if allowed else 'no'}. Food reachable through empty cells: {'yes' if reachable else 'no'}."
    criteria = {a: "Blocked. Collision." if m["blocked"] else "Unsafe. Traps the snake." if a not in allowed
                else "Safe. Eat food now. Best." if m["eats"] else "Safe. Best route to food." if a == best
                else "Safe. Slower route." for a, m in moves.items()}
    return state, {"move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}


def geometry_input(game, moves):
    """Model only: board geometry and plain direction options, without the planner's verdicts."""
    hx, hy = game.body[0]
    lines = [f"Snake on a {game.size} by {game.size} board. Head ({hx},{hy}), heading {game.direction.upper()}. "
             f"Food ({game.food[0]},{game.food[1]}). Length {len(game.body)}. Coordinates start at the top left."]
    for action, move in moves.items():
        lines.append(f"{action.upper()}: BLOCKED." if move["blocked"] else
                     f"{action.upper()}: OPEN; food distance {move['food_distance']}; open region {move['reachable_cells']} cells.")
    criteria = {a: f"move {a.upper()} one cell" for a in moves}
    return "\n".join(lines), {"move": {"type": "choice", "instructions": RAW_INSTRUCTIONS, "criteria": criteria}}


def greedy_move(game):
    moves = inspect_moves(game)
    return min(moves, key=lambda a: (moves[a]["blocked"], moves[a]["food_distance"], -moves[a]["reachable_cells"]))


def execute_decision(game, model, mode="shield"):
    if mode not in ("shield", "model"):
        raise ValueError("Unknown control mode")
    started = time.perf_counter()
    state = game.state()
    moves = inspect_moves(game)
    # Model + safety gets the planner's verdicts and the shield; Model only gets neither.
    prompt, questions = model_input(moves) if mode == "shield" else geometry_input(game, moves)
    prediction = time.perf_counter()
    response = model.predict(prompt, questions) if model else {"answers": {"move": {"choice": greedy_move(game)}}}
    api_ms = (time.perf_counter()-prediction)*1000
    answer = response["answers"]["move"]
    proposed = answer["choice"]
    if proposed not in moves:
        raise ValueError("Model returned an unknown direction")
    action = proposed
    if mode == "shield" and not moves[proposed]["shield_allowed"]:
        allowed = [a for a in moves if moves[a]["shield_allowed"]]
        if not allowed:
            raise RuntimeError("Cycle safety invariant failed; no safe move")
        probabilities = answer.get("probabilities", {})
        action = max(allowed, key=lambda a: (probabilities.get(a, 0), -moves[a]["food_distance"]))
    intervention = action != proposed
    game.step(action)
    return {"state": state, "after": game.state(), "model_state": prompt, "questions": questions, "answers": response["answers"],
            "usage": response.get("usage"), "metrics": model.last_metrics.copy() if model else {},
            "action": action, "proposed": proposed, "intervention": intervention, "mode": mode,
            "shield_reason": moves[proposed]["reason"] if intervention else None, "candidates": moves,
            "ms": api_ms, "step_ms": (time.perf_counter()-started)*1000}


def replay_html(episodes, path):
    data = json.dumps(episodes, ensure_ascii=False).replace("<", "\\u003c")
    page = Path(__file__).with_name("snake_replay.html").read_text(encoding="utf-8")
    path.write_text(page.replace("__EPISODES__", data), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cuda", "official", "upstream", "jev", "greedy"], default="cuda")
    parser.add_argument("--checkpoint", default="models/laya-multilingual")
    parser.add_argument("--mode", choices=["shield", "model"], default="shield")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("reports/snake"))
    parser.add_argument("--export-cases", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1 or args.output.exists() or args.size < 6 or args.size % 2:
        parser.error("Use positive steps, an even board size >= 6, and a fresh output directory")
    args.output.mkdir(parents=True)
    model = None if args.backend == "greedy" else open_backend(args.backend, args.checkpoint,
                                                             max_requests=len(args.seeds)*args.max_steps)
    episodes, cases = [], []
    try:
        for seed in args.seeds:
            game, frames = Snake(seed, args.size), []
            while not game.done and game.steps < args.max_steps:
                gold = greedy_move(game) if args.export_cases else None
                frame = execute_decision(game, model, args.mode)
                frames.append(frame)
                if args.export_cases:
                    cases.append({"id": f"snake/{seed}/{game.steps-1}", "suite": "snake.greedy_agreement",
                                  "state": frame["model_state"], "questions": frame["questions"],
                                  "gold": {"move": {"label": gold}},
                                  "label_source": "greedy geometry heuristic; not optimal-action ground truth"})
            episode = {"backend": args.backend, "checkpoint": args.checkpoint, "mode": args.mode,
                       "seed": seed, "score": game.score, "steps": game.steps,
                       "reason": game.reason or "step_limit", "frames": frames,
                       "interventions": sum(f["intervention"] for f in frames),
                       "note": "Planner verdicts in option text. Safety corrections are rules, not model decisions. Different policies produce different trajectories."}
            episodes.append(episode)
            (args.output/"episodes.json").write_text(json.dumps(episodes, indent=2), encoding="utf-8")
            print({k: v for k, v in episode.items() if k not in ("frames", "note")}, flush=True)
    finally:
        if model:
            model.close()
    replay_html(episodes, args.output/"replay.html")
    if args.export_cases:
        args.export_cases.parent.mkdir(parents=True, exist_ok=True)
        args.export_cases.write_text("".join(json.dumps(row)+"\n" for row in cases), encoding="utf-8")


if __name__ == "__main__":
    main()
