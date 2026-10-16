"""Part 7: measure the core, not just pass or fail.

The check runner from Part 2 answers "does this case pass?". Measuring asks
how well each stage works and where it is weak:

- Retrieval, per question: did a relevant block arrive (hit), how many of the
  relevant blocks arrived (recall), how much of what arrived is relevant
  (precision), and how high the first relevant block ranked.
- The same numbers grouped by kind of question, so one average can't hide a
  weak spot.
- A held-out set of questions nobody tuned the system on.
- Stability: do rewordings of a question get the same result?
- Size: roughly how many tokens each prompt costs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from atlas.checks import CASES_FILE, evaluate
from atlas.pipeline import Assistant, Trace

HELD_OUT_FILE = CASES_FILE.parent / "held_out.json"


def approx_tokens(text: str) -> int:
    """About four characters per token for English text: a rough rule, not a tokenizer."""
    return max(1, len(text) // 4)


@dataclass
class Row:
    case_id: str
    kind: str
    status: str
    hit: float | None
    recall: float | None
    precision: float | None
    first_relevant: int | None
    blocks: int
    tokens: int


def retrieval_row(case: dict, trace: Trace) -> Row:
    # part-09: records from the list path can share a source section; count each section once
    supplied = list(dict.fromkeys(block.ref for block in trace.context))
    relevant = set(case.get("relevant", []))
    found = relevant & set(supplied)
    if relevant:
        hit = 1.0 if found else 0.0
        recall = len(found) / len(relevant)
        precision = len(found) / len(supplied) if supplied else 0.0
        first = next((i for i, ref in enumerate(supplied, start=1) if ref in relevant), None)
    else:  # nothing in the documents answers it: retrieval can't be scored
        hit = recall = precision = first = None
    return Row(case["id"], case.get("kind", "?"), trace.status, hit, recall, precision, first,
               len(supplied), approx_tokens(trace.prompt) if trace.prompt else 0)


def measured_cases(path: Path) -> list[dict]:
    """One case per distinct question: the ones annotated with 'relevant'."""
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cases"] if "relevant" in c]


def measure(assistant: Assistant, path: Path) -> list[Row]:
    return [retrieval_row(case, assistant.ask(case["question"])) for case in measured_cases(path)]


def mean(values: list[float | None]) -> float | None:
    scored = [v for v in values if v is not None]
    return sum(scored) / len(scored) if scored else None


def by_kind(rows: list[Row]) -> dict[str, dict[str, float | None]]:
    kinds: dict[str, list[Row]] = {}
    for row in rows:
        kinds.setdefault(row.kind, []).append(row)
    return {
        kind: {
            "questions": len(group),
            "hit": mean([r.hit for r in group]),
            "recall": mean([r.recall for r in group]),
            "precision": mean([r.precision for r in group]),
        }
        for kind, group in kinds.items()
    }


def stability(assistant: Assistant, path: Path) -> list[tuple[str, str, bool, str]]:
    """Run each case's own checks on its rewordings."""
    results = []
    for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
        for variant in case.get("variants", []):
            outcomes = [evaluate(assistant.ask(variant), check) for check in case["checks"]]
            results.append((case["id"], variant, all(ok for ok, _ in outcomes), "; ".join(d for _, d in outcomes)))
    return results


def export_ragas(assistant: Assistant, path: Path, out: Path) -> int:
    """Write one JSON line per question with the field names Ragas expects."""
    lines = []
    for case in measured_cases(path):
        trace = assistant.ask(case["question"])
        lines.append(json.dumps({
            "user_input": case["question"],
            "retrieved_contexts": [block.text for block in trace.context],
            "response": trace.answer,
            "reference": case.get("behavior", ""),
        }))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)
