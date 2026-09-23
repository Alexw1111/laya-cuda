"""Run a decision with the installed library: uv run python examples/predict.py."""
from laya_cuda import Engine


def main():
    questions = {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {"billing": "payments and refunds", "technical": "software errors"},
        }
    }
    with Engine("laya") as model:
        result = model.predict("Please refund my duplicate payment.", questions)
        print(result["answers"])


if __name__ == "__main__":
    main()
