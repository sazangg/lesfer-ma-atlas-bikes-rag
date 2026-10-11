"""Command line: `python -m atlas ask "question" [--show-prompt]`."""

from __future__ import annotations

import argparse

from atlas import CHECKPOINT
from atlas.pipeline import default_assistant


def main() -> None:
    parser = argparse.ArgumentParser(prog="atlas", description=f"Atlas assistant ({CHECKPOINT})")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    if args.command == "ask":
        trace = default_assistant().ask(args.question)
        print(f"[{CHECKPOINT}] {trace.question}\n\nRetrieved:")
        for chunk_id, score in trace.retrieved:
            print(f"  {score:5.3f}  {chunk_id}")
        if args.show_prompt:
            print(f"\nPrompt:\n{trace.prompt}")
        print(f"\nAnswer ({trace.status}):\n{trace.answer}")


if __name__ == "__main__":
    main()
