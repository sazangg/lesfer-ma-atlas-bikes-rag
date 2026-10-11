"""Command line.

    python -m atlas ask "question" [--show-prompt]
    python -m atlas check [--save-traces]
"""

from __future__ import annotations

import argparse
import sys

from atlas import CHECKPOINT
from atlas.pipeline import default_assistant


def main() -> None:
    parser = argparse.ArgumentParser(prog="atlas", description=f"Atlas assistant ({CHECKPOINT})")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--show-prompt", action="store_true")
    check = commands.add_parser("check", help="run the Atlas cases against this checkpoint")
    check.add_argument("--save-traces", action="store_true", help="write each trace to traces/")
    args = parser.parse_args()

    if args.command == "ask":
        trace = default_assistant().ask(args.question)
        print(f"[{CHECKPOINT}] {trace.question}\n\nRetrieved:")
        for chunk_id, score in trace.retrieved:
            print(f"  {score:5.3f}  {chunk_id}")
        if args.show_prompt:
            print(f"\nPrompt:\n{trace.prompt}")
        print(f"\nAnswer ({trace.status}):\n{trace.answer}")

    elif args.command == "check":
        from atlas.checks import run

        results = run(default_assistant(), save_traces=args.save_traces)
        print(f"Checkpoint {CHECKPOINT}\n")
        for r in results:
            print(f"  {r.label:<22} {r.case_id:<24} {r.detail}")
        regressions = sum(r.label == "REGRESSION" for r in results)
        gaps = sum(r.label.startswith("KNOWN GAP") for r in results)
        print(f"\n{len(results) - regressions - gaps} passing, {gaps} known gaps, {regressions} regressions")
        sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
