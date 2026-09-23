"""Three decision workflows using either local Laya or Jev through OpenRouter."""
import argparse
import json
from benchmarks.backends import open_backend

WORKFLOWS = {
    "triage": ({"message": "I was charged twice. Please refund the duplicate today."}, {
        "team": {"type": "choice", "instructions": "Which team should handle this request?",
                 "criteria": {"billing": "charges, payments and refunds", "technical": "software bugs", "sales": "buying a product"}},
        "urgent": {"type": "noul", "instructions": "Does the customer request action today?"}}),
    "routing": ({"request": "Explain why this SQL query produces duplicate rows and fix the join."}, {
        "route": {"type": "choice", "instructions": "Choose the appropriate specialist.",
                  "criteria": {"code": "programming and debugging", "writing": "editing prose", "general": "general questions"}}}),
    "relevance": ({"question": "How do I reset my password?", "passage": "Open account settings, select Security, then Reset password."}, {
        "relevance": {"type": "score", "instructions": "How well does the passage answer the question?",
                      "criteria": ["unrelated", "partly relevant", "direct answer"]}}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cuda", "official", "upstream", "jev"], default="cuda")
    parser.add_argument("--checkpoint", default="models/laya")
    parser.add_argument("--workflow", choices=list(WORKFLOWS)+["all"], default="all")
    args = parser.parse_args()
    with open_backend(args.backend, args.checkpoint, max_requests=3) as model:
        for name, (state, questions) in WORKFLOWS.items():
            if args.workflow in (name, "all"):
                print(json.dumps({"workflow": name, "result": model.predict(state, questions)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
