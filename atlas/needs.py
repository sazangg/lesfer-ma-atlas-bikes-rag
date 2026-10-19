"""Part 14: which parts of the series do your questions need?

Every module in this series closes a gap that some kind of question opens.
So the architecture follows from the questions: route a sample of real ones,
count the paths they take, and build the modules your users actually need.

    python -m atlas needs traffic/week-1.json
    python -m atlas needs my-questions.txt      (one question per line)

The router here is the stand-in from part-09: a few rules. Its counts are a
first look, not a verdict. Check a sample by hand, and compare with the review
queue from part-08: a question that search answers badly may need a module
even though the router sent it to search.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from atlas.pipeline import ROUTES, validate_route
from atlas.toy import ToyRouter


def read_questions(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("requests") or data.get("cases") or data.get("questions") or []
        return [item["question"] if isinstance(item, dict) else item for item in items]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def needs(args) -> None:
    questions = read_questions(Path(args.file))
    router = ToyRouter()
    paths = [validate_route(router.route(q))["path"] for q in questions]
    counts = Counter(paths)
    example = {p: q for q, p in reversed(list(zip(questions, paths)))}

    print(f"{len(questions)} questions from {args.file}\n")
    print(f"{'route':<10} {'questions':>9}  {'built in':<26} example")
    for route, part in ROUTES.items():
        n = counts.get(route, 0)
        shown = example.get(route, "")
        shown = shown if len(shown) <= 60 else shown[:57] + "..."
        print(f"{route:<10} {n:>4} {100 * n // max(1, len(questions)):>3}%  {part:<26} {shown}")

    modules = [ROUTES[r] for r in ROUTES if r != "search" and counts.get(r)]
    print("\nModules these questions need: " + ("; ".join(modules) if modules else "none, the core covers them"))
    print("Routed by the stand-in router. Check a sample by hand, and read part-08's review queue too.")
