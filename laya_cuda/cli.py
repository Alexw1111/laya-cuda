"""laya-cuda command line: predict, models, download, doctor.

GPU libraries load only inside commands that need them, so --help and `models` stay fast.
"""
import argparse
from contextlib import contextmanager, nullcontext
import json
import platform
import sys
import time
import traceback
from pathlib import Path

from . import __version__
from .engine import Engine
from .models import _checkpoints, resolve

MIN_DRIVER_CUDA = 12040  # packaged CUDA 12.4.1 components; see README "Platform and dependencies"


class CliError(Exception):
    """An expected failure, reported in one line without a traceback."""


def dump(value, stream, pretty=False):
    # Non-UTF-8 consoles (e.g. a GBK code page) get ASCII escapes, which are still valid JSON.
    ascii_only = (getattr(stream, "encoding", None) or "").lower().replace("-", "") != "utf8"
    stream.write(json.dumps(value, ensure_ascii=ascii_only, indent=2 if pretty else None, default=str) + "\n")


def load_questions(value):
    """Inline JSON, a file path, or - for stdin."""
    try:
        if value.lstrip().startswith("{"):
            return json.loads(value)
        return json.loads(sys.stdin.read() if value == "-" else Path(value).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CliError(f"questions file not found: {value}") from None
    except json.JSONDecodeError as error:
        raise CliError(f"questions are not valid JSON: {error}") from None


def open_engine(args):
    return Engine(args.model, device=args.device, backend=args.backend)


def predict(args):
    questions = load_questions(args.questions) if args.questions else None
    if (args.state is None) == (args.input is None):
        raise CliError("give either a STATE or --input")
    if args.state is not None:
        if questions is None:
            raise CliError("--questions is required")
        state = sys.stdin.read() if args.state == "-" else args.state
        with open_engine(args) as engine:
            result = engine.predict(state, questions)
            if args.metrics:
                result["metrics"] = engine.last_metrics
        with output(args.output) as out:
            dump(result, out, pretty=out is sys.stdout)
        return 0
    try:
        source = nullcontext(sys.stdin) if args.input == "-" else open(args.input, encoding="utf-8")
    except FileNotFoundError:
        raise CliError(f"input file not found: {args.input}") from None
    failed = 0
    with source as lines, output(args.output) as out, open_engine(args) as engine:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            rid = number
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or "state" not in request:
                    raise ValueError('expected an object with a "state" field')
                rid = request.get("id", number)
                if request.get("questions", questions) is None:
                    raise ValueError('no "questions" on this line and no --questions given')
                result = {"id": rid, **engine.predict(request["state"], request.get("questions", questions))}
                if args.metrics:
                    result["metrics"] = engine.last_metrics
            except Exception as error:  # one bad line must not discard the rest of the batch
                failed += 1
                result = {"id": rid, "error": f"{type(error).__name__}: {error}"}
            dump(result, out)
            out.flush()
    if failed:
        print(f"laya-cuda: {failed} request(s) failed; see the \"error\" field", file=sys.stderr)
    return 1 if failed else 0


@contextmanager
def output(path):
    """stdout by default, or the --output file."""
    if path is None:
        yield sys.stdout
        return
    with open(path, "w", encoding="utf-8") as stream:
        yield stream


def models(args):
    from huggingface_hub import try_to_load_from_cache
    entries = _checkpoints()
    width = max(map(len, entries))
    for name, entry in entries.items():
        cached = try_to_load_from_cache(entry["repo_id"], "model.safetensors", revision=entry["revision"])
        print(f"{name:<{width}}  {'downloaded' if isinstance(cached, str) else '-':<10}  "
              f"{entry['repo_id']}@{entry['revision'][:12]}")
    return 0


def download(args):
    print(resolve(args.model))
    return 0


def doctor(args):
    checks = []

    def report(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})
        if not args.json:
            print(f"  {status:<4}  {name}: {detail}", flush=True)

    def run():
        try:
            from .runtime import Ops, cp
            from .ops import MIN_ARCH
            from cuda.pathfinder import load_nvidia_dynamic_lib
        except Exception as error:
            return report("CuPy", "fail", f"{type(error).__name__}: {error}")
        report("CuPy", "ok", cp.__version__)
        driver = cp.cuda.runtime.driverGetVersion()
        text = f"supports CUDA {driver // 1000}.{driver % 1000 // 10}"
        if driver < 12000:
            return report("driver", "fail", f"{text}; CUDA 12 is required")
        report("driver", "ok" if driver >= MIN_DRIVER_CUDA else "warn",
               text if driver >= MIN_DRIVER_CUDA else f"{text}; tested with 12.4+ (Windows 551.78+, Linux 550.54.15+)")
        if not cp.cuda.runtime.getDeviceCount():
            return report("GPU", "fail", "no CUDA device found")
        for index in range(cp.cuda.runtime.getDeviceCount()):
            p = cp.cuda.runtime.getDeviceProperties(index)
            detail = f"{p['name'].decode()}, sm_{p['major']}{p['minor']}, {p['totalGlobalMem'] / 2**30:.1f} GiB"
            supported = (p["major"], p["minor"]) >= MIN_ARCH
            report(f"GPU {index}", "ok" if supported else "warn",
                   detail if supported else f"{detail}; unsupported, compute capability 8.0+ (Ampere or later) is required")
        report("NVRTC", "ok", ".".join(map(str, cp.cuda.nvrtc.getVersion())))
        report("cuBLAS", "ok", load_nvidia_dynamic_lib("cublas").abs_path)
        started = time.perf_counter()
        with cp.cuda.Device(int(args.device.rpartition(":")[2] or 0)):
            Ops(cp.cuda.Stream(non_blocking=True)).close()
        report("kernels", "ok", f"ready in {(time.perf_counter()-started)*1000:.0f} ms")
        if args.model:
            questions = {"urgent": {"type": "noul", "instructions": "Does this need urgent attention?"}}
            started = time.perf_counter()
            with open_engine(args) as engine:
                load, times = time.perf_counter() - started, []
                for _ in range(3):
                    started = time.perf_counter()
                    engine.predict("Please refund my duplicate payment.", questions)
                    times.append((time.perf_counter() - started) * 1000)
            report(f"model {args.model}", "ok", f"load {load:.2f} s, first {times[0]:.1f} ms, warm {min(times[1:]):.1f} ms")

    if not args.json:
        print(f"laya-cuda {__version__}, Python {platform.python_version()}, {platform.platform(terse=True)}")
    try:
        run()
    except Exception as error:
        report("error", "fail", f"{type(error).__name__}: {error}")
    passed = all(c["status"] != "fail" for c in checks)
    if args.json:
        dump({"version": __version__, "platform": platform.platform(), "passed": passed, "checks": checks}, sys.stdout, True)
    else:
        print("All checks passed." if passed else "Some checks failed.")
    return 0 if passed else 1


def build_parser():
    parser = argparse.ArgumentParser(prog="laya-cuda", description="Local GPU inference for Laya decision models.")
    parser.add_argument("--version", action="version", version=f"laya-cuda {__version__}")
    parser.add_argument("--debug", action="store_true", help="show tracebacks")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def engine_options(p):
        p.add_argument("--backend", choices=("cuda", "official"), default="cuda", help="default: cuda")
        p.add_argument("--device", default="cuda:0", help="default: cuda:0")

    p = commands.add_parser("predict", help="answer questions about a state", formatter_class=argparse.RawDescriptionHelpFormatter,
                            epilog='examples:\n'
                                   '  laya-cuda predict "Please refund my payment." -q questions.json\n'
                                   '  laya-cuda predict -i requests.jsonl -q questions.json -o results.jsonl\n\n'
                                   'Each JSONL line is {"id": ..., "state": ..., "questions": {...}}; "id" and\n'
                                   '"questions" are optional. Failed lines get an "error" field and exit status 1.')
    p.add_argument("state", nargs="?", metavar="STATE", help="state text, or - to read stdin")
    p.add_argument("-q", "--questions", metavar="JSON", help="questions: a file, inline JSON, or -")
    p.add_argument("-m", "--model", default="laya", help="model name or checkpoint folder (default: laya)")
    p.add_argument("-i", "--input", metavar="JSONL", help="batch of requests, one per line (- for stdin)")
    p.add_argument("-o", "--output", metavar="FILE", help="write results to FILE instead of stdout")
    p.add_argument("--metrics", action="store_true", help="include timing metrics")
    engine_options(p)
    p.set_defaults(handler=predict)

    commands.add_parser("models", help="list model names and download status").set_defaults(handler=models)

    p = commands.add_parser("download", help="download a model's weights ahead of first use")
    p.add_argument("model", metavar="MODEL", help="model name, e.g. laya")
    p.set_defaults(handler=download)

    p = commands.add_parser("doctor", help="check driver, GPU, CUDA libraries and kernels")
    p.add_argument("model", nargs="?", metavar="MODEL", help="also load this model and time a prediction")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    engine_options(p)
    p.set_defaults(handler=doctor)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("laya-cuda: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        if args.debug:
            traceback.print_exc()
        print(f"laya-cuda: error: {error if isinstance(error, CliError) else f'{type(error).__name__}: {error}'}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
