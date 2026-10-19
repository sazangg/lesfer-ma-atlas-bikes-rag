"""Part 13: answer caching.

Whole documents make prompts long, and long prompts cost money and time.
Part 8's log showed the same questions asked again and again. An answer
cache returns a stored answer for a repeated question without retrieving or
calling a model.

The key decides what counts as "the same question". It must include
everything that changes the answer:

    the question, normalized    "What does E417 mean?" == "what does e417 mean"
    the user's access groups    staff and sales may get different answers
    the settings                another k or model can change the answer
    the collection version      a revised document invalidates every entry

Only answered requests are stored: a clarification or a refusal should be
asked again, not replayed.
"""

from __future__ import annotations

import hashlib
import json
import re


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower()).strip(" ?!.")


class AnswerCache:
    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}
        self.hits = 0

    def key(self, question: str, groups: set[str], settings: dict, collection_version: str) -> str:
        material = [normalize(question), sorted(groups), settings, collection_version]
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        entry = self.entries.get(key)
        if entry is not None:
            self.hits += 1
        return entry

    def forget(self, key: str) -> None:
        """A thumbs down evicts the answer, so the next asker gets a fresh one, not a replay of a bad one."""
        self.entries.pop(key, None)

    def put(self, key: str, trace) -> None:
        self.entries[key] = {name: getattr(trace, name) for name in (
            "fields", "retrieved", "context", "citations", "raw_answer", "answer", "status", "route", "complete_by")}
