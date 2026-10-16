"""Run the Atlas cases against the current checkpoint.

Each case says which checkpoint should first pass it. The runner reports:

    PASS        the case passes, as expected
    KNOWN GAP   the case fails and a later part is meant to close it
    REGRESSION  the case fails although this checkpoint should pass it
    EARLY PASS  the case passes before the part meant to close it

Known gaps are not failures of the checkpoint. They are the reason for the
next article.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from atlas import CHECKPOINT
from atlas.pipeline import Assistant, Trace

CASES_FILE = Path(__file__).resolve().parent.parent / "cases" / "cases.json"
TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


def part_number(checkpoint: str) -> int:
    return int(checkpoint.removeprefix("part-"))


def evaluate(trace: Trace, check: dict) -> tuple[bool, str]:
    refs = [chunk.ref for chunk in trace.context]
    supplied = "\n".join(chunk.text for chunk in trace.context)
    kind = check["type"]
    if kind == "includes_refs":
        missing = [r for r in check["refs"] if r not in refs]
        return not missing, f"missing {missing}" if missing else "all present"
    if kind == "excludes_refs":
        present = [r for r in check["refs"] if r in refs]
        return not present, f"also supplied {present}" if present else "none supplied"
    if kind == "first_ref":
        first = refs[0] if refs else None
        return first == check["ref"], f"first is {first}"
    if kind == "includes_text":
        ok = check["text"] in supplied
        return ok, f"'{check['text']}' {'supplied' if ok else 'not supplied'}"
    if kind == "includes_ids":
        found = set(re.findall(r"\bR\d{2}\b", supplied))
        missing = [i for i in check["ids"] if i not in found]
        return not missing, f"missing {missing}" if missing else "all present"
    if kind == "same_block":
        ok = any(all(s in c.text for s in check["contains"]) for c in trace.context)
        return ok, "together in one passage" if ok else "not in the same passage"
    if kind == "block_with":
        blocks = [c.text for c in trace.context if check["contains"] in c.text]
        if not blocks:
            return False, f"{check['contains']} not supplied"
        ok = all(check["also"] in b for b in blocks)
        return ok, f"{check['contains']} supplied {'with' if ok else 'without'} '{check['also']}'"
    if kind == "status":
        return trace.status == check["equals"], f"status {trace.status}"
    if kind == "answer_contains":
        ok = check["text"] in trace.answer
        return ok, "found in answer" if ok else "not in answer"
    if kind == "route":
        return trace.route == check["equals"], f"route {trace.route}"
    if kind == "complete_by":
        complete_by = getattr(trace, "complete_by", None)
        ok = complete_by == check["route"]
        return ok, "completeness guaranteed" if ok else "completeness not guaranteed"
    raise ValueError(f"unknown check type {kind}")


@dataclass
class Result:
    case_id: str
    label: str
    detail: str


def run(assistant: Assistant, save_traces: bool = False) -> list[Result]:
    current = part_number(CHECKPOINT)
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    results = []
    for case in cases:
        trace = assistant.ask(case["question"])
        outcomes = [evaluate(trace, check) for check in case["checks"]]
        passed = all(ok for ok, _ in outcomes)
        due = part_number(case["expected_from"])
        if passed:
            label = "PASS" if current >= due else "EARLY PASS"
        else:
            label = "REGRESSION" if current >= due else f"KNOWN GAP -> {case['expected_from']}"
        results.append(Result(case["id"], label, "; ".join(d for _, d in outcomes)))
        if save_traces:
            _save(case["id"], trace)
    return results


def _save(case_id: str, trace: Trace) -> None:
    directory = TRACES_DIR / CHECKPOINT
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "question": trace.question,
        "retrieved": trace.retrieved,
        "context": [{"ref": c.ref, "id": c.id, "text": c.text} for c in trace.context],
        "prompt": trace.prompt,
        "answer": trace.answer,
        "status": trace.status,
    }
    (directory / f"{case_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
