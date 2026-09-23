# Local library and examples

Nothing is published. Build the wheel locally:

```powershell
uv build
pip install .\dist\laya_cuda-0.1.0-py3-none-any.whl
```

The wheel contains `laya_cuda` and its CUDA kernels. It does not contain examples,
benchmark tools, downloaded weights, research prototypes or an HTTP server.
Run the examples below from the repository root with the uv environment. The
source archive includes them for development.

```python
from laya_cuda import Engine

questions = {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}
with Engine("laya") as model:
    print(model.predict("Please refund the duplicate payment.", questions))
```

`Engine` is synchronous and not thread-safe. Use the opt-in `BatchEngine` for
concurrent callers; it batches through one engine-owning worker. The optional
official backend requires the reference extra. Model weights download separately.

## Decision workflows

```powershell
uv run --no-sync python -m examples.workflows --backend cuda
uv run --no-sync python -m examples.workflows --backend upstream
```

The three examples cover ticket triage, specialist routing and passage relevance.
The upstream command needs the pinned checkout described in the benchmark guide.

## Snake Lab

Start the real local playground (CUDA only; no cloud requests or UI dependencies):

```powershell
uv run --no-sync python -m examples.snake_web --port 8767
```

Open `http://127.0.0.1:8767/` (or `http://localhost:8767/`). Both Snake entry points default to `models/laya-multilingual`;
pass `--checkpoint models/laya` for the English checkpoint. The model loads once on a dedicated owning worker.
Run/pause, single-step, seeded reset, board size (12–24), a pace slider (1–240 moves/s or unlimited, the default), mode switching and JSON export are
interactive. Mode switching restarts the same seed to preserve the cycle invariant.
A run stops at a terminal state or after size⁴ moves (20,736 on 12×12), enough for a
guarded run to fill the board. Memory keeps every latency sample but only the most
recent 500–1,000 decisions; exports report how many earlier frames were omitted. The service binds only to loopback;
stop its terminal with Ctrl+C to release the model. This is an example, not a
production inference server, and is excluded from the installed core wheel.

The model chooses absolute **up / down / left / right**. The cycle planner writes
its verdict into each option's text (`Blocked. Collision.`,
`Unsafe. Traps the snake.`, `Safe. Best route to food.`,
`Safe. Eat food now. Best.` or `Safe. Slower route.`), because Laya scores each
option at its own position. The state holds two planner facts only. The model
reads the planner's recommendation; this is not a test of inferring strategy
from the raw board. Option order matters: on recorded frames, the order
up/right/down/left led to 22 blocked choices in 1,015 decisions, while
up/down/left/right led to none.

The default **Model + safety** mode preserves order on a Hamiltonian cycle. It
allows forward shortcuts only before the tail and without skipping the food.
When the proposed move violates that constraint, the highest-probability allowed
move is executed. Every correction, reason, original probability distribution and
executed action is recorded and displayed. Region size is a geometry measurement,
not a model-estimated dead-end probability. **Model only** applies the proposed
move unchanged, including collisions. It also removes the planner's help: the model sees only board
geometry (coordinates, blocked directions, food distance and open space) and plain direction options,
so it shows what the checkpoint does unaided.

Each step calls the actual model once. Prediction API latency includes input
preparation, synchronized inference and output decoding; full-step latency also
includes geometry and safety. Model loading, browser rendering and pacing are
excluded from both. P50/P95 include first-use costs and describe the recorded
samples, not a benchmark guarantee. Observed game rate includes pacing; it is not
the inverse of inference latency. Changing pace may change measured latency.

Generate portable recordings with the same game and decision implementation:

```powershell
uv run --no-sync python -m examples.snake --backend cuda --mode shield --seeds 0 1 2 --max-steps 300 --output reports/snake-shield
uv run --no-sync python -m examples.snake --backend cuda --mode model --seeds 0 1 2 --max-steps 300 --output reports/snake-pure
uv run --no-sync python -m examples.snake --backend upstream --mode shield --seeds 0 1 2 --output reports/snake-upstream
```

`episodes.json` contains the exact model state, pre/post-move board, probabilities,
corrections, API/GPU/full-step timing and usage. `replay.html` embeds that data and
works offline. Playback controls affect recorded frames, never inference or safety
mode. Downloading a live run exports JSON; turn it into an offline replay with
`examples.snake.replay_html`.

Boards must have even sizes of at least six; the default is 12. Both modes use the
same starting state. Equal seeds do not force equal later trajectories or food
locations. `--export-cases` freezes model inputs for `benchmarks.compare`, using a
clearly labeled greedy geometry heuristic as reference, not optimal-action truth.
The separate `--backend greedy` is a rule baseline, never an AI result.

## Jev through OpenRouter

Jev uses the Decisions API, **not** `/chat/completions`. Official references:
[model](https://openrouter.ai/typesafe/jev-1.13),
[Decisions example](https://github.com/openrouterteam/docs/blob/main/cookbook/building-agents/gate-tool-calls-with-jev.mdx).

Set `OPENROUTER_API_KEY` in your environment, then run:

```powershell
uv run --no-sync python -m examples.workflows --backend jev
uv run --no-sync python -m examples.snake --backend jev --seeds 0 1 2 --max-steps 40 --output reports/snake-jev
```

These commands make paid remote calls. Requests use `typesafe/jev-1.13`; each
response's resolved model, usage and provider metadata are retained. The client
reuses its HTTPS connection, has a timeout, makes no automatic retries and limits
request count. The reported-cost stop is checked before the next request; it is
not a hard billing cap and cannot account for absent cost fields. Local CUDA
latency and remote service latency include different deployment overheads.

No API key is required for local examples, offline tests or viewing replays.
