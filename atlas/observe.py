"""Part 8: watching the system once real users arrive.

Evaluation (Part 7) runs questions we chose. Production brings questions
nobody anticipated, and a wrong answer there looks like any other answer: no
crash, no error. To find those, every request leaves a record:

- a span per stage: what went in, what came out, how long it took;
- one line per request in a log file (JSON lines), with the settings and
  stand-ins in use, so a request can be explained later;
- the user as a pseudonym, not a name: logs are sensitive data too.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "requests.jsonl"


@contextmanager
def span(trace, stage: str, **details):
    """Time one stage and keep its details on the trace. `trace` may be None."""
    record = {"stage": stage, **details}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["ms"] = round((time.perf_counter() - start) * 1000, 3)
        if trace is not None:
            trace.spans.append(record)


def pseudonym(name: str) -> str:
    return "u-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]


def log_record(trace, user, settings: dict) -> dict:
    finite = [score for _, score in trace.retrieved if score != float("inf")]
    return {
        "id": trace.request_id,
        "at": trace.at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": pseudonym(user.name),
        "groups": sorted(user.groups),
        "question": trace.question,
        "status": trace.status,
        "route": trace.route,  # part-09
        "fields": trace.fields,
        "retrieved": [[cid, None if score == float("inf") else score] for cid, score in trace.retrieved],
        "exact_match": any(score == float("inf") for _, score in trace.retrieved),
        "top_score": max(finite) if finite else None,
        "blocks": [block.ref for block in trace.context],
        "prompt_tokens": max(1, len(trace.prompt) // 4) if trace.prompt else 0,
        "citations": trace.citations,
        "answer": trace.answer,
        "spans": trace.spans,
        "settings": settings,
    }


def append(record: dict, path: Path = LOG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_log(path: Path = LOG_FILE) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# Reading the log -------------------------------------------------------------------------

WEAK_SCORE = 0.5  # a best reranked score below this, with no exact match, is a weak search


def requests_and_feedback(entries: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Feedback arrives as separate events, after the answer was shown."""
    requests = [e for e in entries if e.get("type") != "feedback"]
    feedback = {e["id"]: e for e in entries if e.get("type") == "feedback"}
    return requests, feedback


def review_reasons(record: dict, feedback: dict | None) -> list[str]:
    """Why a person should look at this request. Empty if nothing stands out."""
    reasons = []
    if feedback and feedback.get("rating") == "down":
        reasons.append("thumbs down")
    if record["status"] == "clarify":
        reasons.append("asked for clarification")
    elif record["status"] in ("format_error", "number_mismatch"):
        reasons.append("answer withheld")
    elif record["status"] == "cannot_compute":  # part-10
        reasons.append("could not compute")
    elif not record["blocks"]:
        reasons.append("nothing supplied")
    elif not record["exact_match"] and (record["top_score"] or 0) < WEAK_SCORE:
        reasons.append(f"weak search ({record['top_score']})")
    return reasons


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))] if ordered else 0.0
