"""Part 8: the operator's commands. Simulate traffic, read the log, follow one request, keep a failure.

    python -m atlas simulate       replay traffic/week-1.json with logging on
    python -m atlas stats          what happened this week, and what deserves a look
    python -m atlas trace r007     one request, stage by stage
    python -m atlas promote r007   a draft test case from a logged request
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from atlas.observe import (
    LOG_FILE, append, percentile, read_log, requests_and_feedback, review_reasons,
)

ROOT = Path(__file__).resolve().parent.parent
TRAFFIC_FILE = ROOT / "traffic" / "week-1.json"
CANDIDATES_FILE = ROOT / "cases" / "candidates.json"


def simulate(args) -> None:
    from atlas.corpus import load_corpus
    from atlas.pipeline import SALES, STAFF, Assistant

    LOG_FILE.unlink(missing_ok=True)
    from atlas.cache import AnswerCache

    assistant = Assistant(log_path=LOG_FILE, cache=AnswerCache())  # part-13: repeated questions reuse answers
    assistant.ingest(load_corpus())
    users = {"staff": STAFF, "sales": SALES}
    requests = json.loads(TRAFFIC_FILE.read_text(encoding="utf-8"))["requests"]
    for r in requests:
        assistant.ask(r["question"], users[r["user"]], request_id=r["id"], at=r["at"])
        if "feedback" in r:
            append({"type": "feedback", "id": r["id"], **r["feedback"]}, LOG_FILE)
            if r["feedback"]["rating"] == "down":
                assistant.forget(r["question"], users[r["user"]])
    print(f"Replayed {len(requests)} requests into {LOG_FILE.relative_to(ROOT)}")


def stats(args) -> None:
    records, feedback = requests_and_feedback(read_log())
    if not records:
        print("The log is empty. Run `python -m atlas simulate` first.")
        return

    days = sorted({r["at"][:10] for r in records})
    print(f"{len(records)} requests from {len({r['user'] for r in records})} users, {days[0]} to {days[-1]}\n")

    outcomes = Counter(r["status"] for r in records)
    nothing = sum(1 for r in records if r["status"] == "answered" and not r["blocks"])
    print("Outcomes:")
    for status, n in outcomes.most_common():
        print(f"  {status:<12} {n:>3}  ({100 * n // len(records)}%)")
    print(f"  answered with nothing supplied: {nothing}")
    routes = Counter(r.get("route", "search") for r in records)
    print("Routes:", ", ".join(f"{route} {n}" for route, n in routes.most_common()))
    cached = sum(1 for r in records if r.get("cached"))
    print(f"Answered from the cache: {cached}")

    reasons = {r["id"]: review_reasons(r, feedback.get(r["id"])) for r in records}
    weak = sum(any(x.startswith("weak search") for x in v) for v in reasons.values())
    unknown = sum(1 for r in records for s in r["spans"] if s["stage"] == "citations" and s.get("unknown_labels"))
    print(f"\nWeak searches (best score < 0.5, no exact match): {weak}")
    print(f"Answers withheld for unknown citation labels:     {unknown}")

    ratings = Counter(f["rating"] for f in feedback.values())
    print(f"Feedback: {ratings.get('up', 0)} up, {ratings.get('down', 0)} down, "
          f"{len(records) - len(feedback)} requests without feedback")

    print("\nTime per stage, milliseconds (median / 95th percentile):")
    stages: dict[str, list[float]] = {}
    for r in records:
        for s in r["spans"]:
            stages.setdefault(s["stage"], []).append(s["ms"])
    for stage, values in sorted(stages.items(), key=lambda kv: -percentile(kv[1], 0.95)):
        print(f"  {stage:<15} {percentile(values, 0.5):8.3f} / {percentile(values, 0.95):8.3f}")
    tokens = [r["prompt_tokens"] for r in records if r["prompt_tokens"]]
    print(f"Prompt size: ~{sum(tokens) // len(tokens)} tokens on average, {max(tokens)} at most")

    products = Counter((r["fields"] or {}).get("product") or "none named" for r in records)
    print("\nProducts named in questions:", ", ".join(f"{p} {n}" for p, n in products.most_common()))
    repeats = [(q, n) for q, n in Counter(r["question"] for r in records).items() if n > 1]
    for q, n in repeats:
        print(f"Asked {n} times: {q!r}")

    print("\nReview queue:")
    for r in records:
        if reasons[r["id"]]:
            print(f"  {r['id']}  {', '.join(reasons[r['id']]):<44} {r['question']}")


def trace(args) -> None:
    records, feedback = requests_and_feedback(read_log())
    record = next((r for r in records if r["id"] == args.request_id), None)
    if record is None:
        print(f"No request {args.request_id} in the log.")
        return
    print(f"{record['id']}  {record['at']}  {record['user']} ({', '.join(record['groups'])})")
    print(f"Question: {record['question']}")
    print(f"Status:   {record['status']}")
    if record["id"] in feedback:
        f = feedback[record["id"]]
        print(f"Feedback: {f['rating']}" + (f" - {f['comment']}" if f.get("comment") else ""))
    print("\nStages:")
    for s in record["spans"]:
        details = {k: v for k, v in s.items() if k not in ("stage", "ms")}
        shown = ", ".join(f"{k}={v}" for k, v in details.items())
        print(f"  {s['stage']:<15} {s['ms']:7.3f} ms  {shown}")
    print(f"\nRetrieved: {record['retrieved']}")
    print(f"Supplied:  {record['blocks']}")
    print(f"Answer:    {record['answer'][:300]}")
    print(f"Settings:  {record['settings']}")


def promote(args) -> None:
    records, feedback = requests_and_feedback(read_log())
    record = next((r for r in records if r["id"] == args.request_id), None)
    if record is None:
        print(f"No request {args.request_id} in the log.")
        return
    data = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8")) if CANDIDATES_FILE.exists() else {
        "about": "Draft test cases promoted from production requests. A person completes each one from the "
                 "source documents (expected behavior, checks, expected_from) before moving it into cases.json.",
        "cases": [],
    }
    if any(c["from_request"] == record["id"] for c in data["cases"]):
        print(f"{record['id']} is already a candidate.")
        return
    data["cases"].append({
        "id": f"P-{record['id']}",
        "from_request": record["id"],
        "question": record["question"],
        "observed": {"status": record["status"], "supplied": record["blocks"],
                     "why_flagged": review_reasons(record, feedback.get(record["id"]))},
        "behavior": "TODO: write the expected behavior from the source documents",
        "checks": [],
    })
    CANDIDATES_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Added P-{record['id']} to {CANDIDATES_FILE.relative_to(ROOT)}. Complete it before moving it into cases.json.")
