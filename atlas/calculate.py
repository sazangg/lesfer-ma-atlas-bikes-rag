"""Part 10: when the answer must be calculated.

A model reading a table of per-batch rates is tempted to average them. For
batches of different sizes that is wrong: the pass rate of batches A and B is
(80 + 9) / (100 + 10) = 80.9%, not (80% + 90%) / 2 = 85%.

So a metric is defined once, in code, and computed from records:

    metric      records      numerator   denominator   definition
    pass_rate   inspection   passed      inspected     QA-INSPECT-2026, section 1

The model receives the result with its counts and sources, and explains it.
It never does the arithmetic. If an input is missing, there is no result.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.chunking import Chunk
from atlas.index import readable
from atlas.situation import Situation


@dataclass(frozen=True)
class Metric:
    label: str
    record_type: str
    numerator: str
    denominator: str
    counts: str  # what the numerator and denominator count, for the result sentence
    definition: str  # the document section that defines the metric


METRICS = {
    "pass_rate": Metric(
        label="pass rate", record_type="inspection", numerator="passed", denominator="inspected",
        counts="bikes passed final inspection without rework", definition="QA-INSPECT-2026#1",
    ),
}


def calculate(name: str, records: list[Chunk], groups: set[str], situation: Situation,
              batches: list[str] | None = None) -> dict:
    """Compute a metric over the records in scope. Returns a status and, only if it is "ok", a value."""
    metric = METRICS[name]
    rows = [r for r in records
            if r.meta["record_type"] == metric.record_type and readable(r, groups) and situation.allows(r)]
    if batches:
        unknown = sorted(set(batches) - {r.meta["batch"] for r in rows})
        if unknown:
            return {"status": "unknown_scope", "metric": name, "unknown": unknown}
        rows = [r for r in rows if r.meta["batch"] in batches]
    if not rows:
        return {"status": "no_data", "metric": name}

    # A blank is not a zero: one missing count means there is no result, not a smaller one.
    incomplete = [r.meta["batch"] for r in rows
                  if not (r.meta.get(metric.numerator, "").isdigit() and r.meta.get(metric.denominator, "").isdigit())]
    if incomplete:
        return {"status": "incomplete_data", "metric": name, "incomplete": incomplete, "records": rows}

    numerator = sum(int(r.meta[metric.numerator]) for r in rows)
    denominator = sum(int(r.meta[metric.denominator]) for r in rows)
    if denominator == 0:
        return {"status": "no_data", "metric": name}
    return {
        "status": "ok", "metric": name, "scope": [r.meta["batch"] for r in rows],
        "numerator": numerator, "denominator": denominator,
        "value": round(100 * numerator / denominator, 1), "records": rows,
    }


def result_sentence(result: dict) -> str:
    metric = METRICS[result["metric"]]
    return (f"{result['numerator']} of {result['denominator']} {metric.counts}: "
            f"a {metric.label} of {result['value']}% (batches {', '.join(result['scope'])}).")
