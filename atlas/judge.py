"""Part 7: judging answers, and checking the judge.

Evidence checks can't tell whether an answer's wording is right. For that,
teams use a judge: usually another language model asked "does this answer
say only what the sources support?". A judge is itself a model making
judgments, so before trusting it, compare its verdicts with labels a person
gave the same answers.

ToyJudge is a stand-in: a sentence counts as supported when most of its words
appear in the supplied sources. It fails in instructive ways, which is the
point of calibrating it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from atlas.checks import CASES_FILE
from atlas.keyword import terms
from atlas.pipeline import Assistant

JUDGED_FILE = CASES_FILE.parent / "judged_answers.json"


class ToyJudge:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold  # share of a sentence's words that must appear in the sources

    def verdict(self, answer: str, sources: list[str]) -> str:
        source_terms = set(terms(" ".join(sources)))
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
            words = terms(sentence)
            if words and sum(w in source_terms for w in words) / len(words) < self.threshold:
                return "wrong"
        return "correct"


@dataclass
class Calibration:
    answer_id: str
    human: str
    judge: str
    note: str


def calibrate(assistant: Assistant, judge: ToyJudge, path: Path = JUDGED_FILE) -> list[Calibration]:
    results = []
    for item in json.loads(path.read_text(encoding="utf-8"))["answers"]:
        sources = [block.text for block in assistant.ask(item["question"]).context]
        results.append(Calibration(item["id"], item["human"], judge.verdict(item["answer"], sources), item["note"]))
    return results
