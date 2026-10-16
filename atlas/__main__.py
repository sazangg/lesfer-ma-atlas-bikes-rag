"""Command line.

    python -m atlas ask "question" [--show-prompt]
    python -m atlas check [--save-traces]
    python -m atlas ingest-report [--no-corrections]
    python -m atlas measure [--held-out] [--min-score X] [--export-ragas FILE]
    python -m atlas judge-calibration
    python -m atlas simulate | stats | trace ID | promote ID
    python -m atlas records
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
    report = commands.add_parser("ingest-report", help="show what conversion changed and what needs review")
    report.add_argument("--no-corrections", action="store_true", help="ignore corpus/corrections.json")
    measure = commands.add_parser("measure", help="retrieval metrics, results by kind, stability")
    measure.add_argument("--held-out", action="store_true", help="measure the held-out questions instead")
    measure.add_argument("--min-score", type=float, default=None, help="try a different minimum relevance score")
    measure.add_argument("--export-ragas", metavar="FILE", help="write the questions as a Ragas dataset (JSON lines)")
    commands.add_parser("judge-calibration", help="compare the judge's verdicts with human labels")
    commands.add_parser("records", help="build the records, check them against the documents, write knowledge/records.json")
    commands.add_parser("simulate", help="replay a week of simulated traffic with logging on")
    commands.add_parser("stats", help="summarize the request log")
    trace_cmd = commands.add_parser("trace", help="show one logged request stage by stage")
    trace_cmd.add_argument("request_id")
    promote = commands.add_parser("promote", help="turn a logged request into a draft test case")
    promote.add_argument("request_id")
    args = parser.parse_args()

    if args.command in ("simulate", "stats", "trace", "promote"):
        from atlas import ops

        getattr(ops, args.command)(args)
        return

    if args.command == "records":
        from collections import Counter

        from atlas.records import RECORDS_FILE, completeness_report, save

        assistant = default_assistant()
        counts = Counter(r.meta["record_type"] for r in assistant.records)
        curated = sum(r.meta.get("curated") == "yes" for r in assistant.records)
        print("Records: " + ", ".join(f"{n} {t}" for t, n in counts.items()) + f" ({curated} curated by a person)")
        for line in completeness_report(assistant.documents, assistant.records):
            print(f"  {line}")
        save(assistant.records)
        print(f"Wrote {RECORDS_FILE.relative_to(RECORDS_FILE.parent.parent)}")
        return

    if args.command == "measure":
        _measure(args)
        return

    if args.command == "judge-calibration":
        from atlas.judge import ToyJudge, calibrate

        results = calibrate(default_assistant(), ToyJudge())
        print(f"{'answer':<7} {'person':<8} {'judge':<8} note")
        for r in results:
            flag = "" if r.human == r.judge else "  <- disagree"
            print(f"{r.answer_id:<7} {r.human:<8} {r.judge:<8} {r.note}{flag}")
        agree = sum(r.human == r.judge for r in results)
        print(f"\nAgreement with the person: {agree} of {len(results)}")
        return

    if args.command == "ingest-report":
        from atlas.corpus import load_corpus
        from atlas.pipeline import Assistant

        assistant = Assistant()
        assistant.ingest(load_corpus(), corrections={} if args.no_corrections else None)
        for r in assistant.reports:
            print(f"{r.source_id} ({r.file})")
            for line in r.lines():
                print(f"  - {line}")
        return

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


def _measure(args) -> None:
    from pathlib import Path

    from atlas.checks import CASES_FILE
    from atlas.corpus import load_corpus
    from atlas.measure import HELD_OUT_FILE, by_kind, export_ragas, mean, measure, stability
    from atlas.pipeline import Assistant

    assistant = Assistant() if args.min_score is None else Assistant(min_score=args.min_score)
    assistant.ingest(load_corpus())
    path = HELD_OUT_FILE if args.held_out else CASES_FILE

    if args.export_ragas:
        count = export_ragas(assistant, path, Path(args.export_ragas))
        print(f"Wrote {count} questions to {args.export_ragas}")
        return

    def fmt(value) -> str:
        return "  -  " if value is None else f"{value:5.2f}"

    rows = measure(assistant, path)
    print(f"{'held-out' if args.held_out else 'development'} questions, min score {assistant.min_score}\n")
    print(f"{'case':<24} {'kind':<20} {'hit':>5} {'recall':>6} {'prec.':>6} {'rank':>4} {'blocks':>6} {'tokens':>6}")
    for r in rows:
        rank = "-" if r.first_relevant is None else str(r.first_relevant)
        print(f"{r.case_id:<24} {r.kind:<20} {fmt(r.hit):>5} {fmt(r.recall):>6} {fmt(r.precision):>6} {rank:>4} {r.blocks:>6} {r.tokens:>6}")
    tokens = [r.tokens for r in rows if r.tokens]
    print(f"\nmean recall {fmt(mean([r.recall for r in rows]))}  mean precision {fmt(mean([r.precision for r in rows]))}"
          f"  mean prompt ~{sum(tokens) // len(tokens)} tokens")

    print("\nBy kind of question:")
    for kind, m in by_kind(rows).items():
        print(f"  {kind:<20} {m['questions']} question(s)  recall {fmt(m['recall'])}  precision {fmt(m['precision'])}")

    if not args.held_out:
        results = stability(assistant, CASES_FILE)
        print(f"\nStability: {sum(ok for _, _, ok, _ in results)} of {len(results)} rewordings keep their result")
        for case_id, variant, ok, detail in results:
            if not ok:
                print(f"  CHANGED  {case_id}: {variant!r} ({detail})")


if __name__ == "__main__":
    main()
