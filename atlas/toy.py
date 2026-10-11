"""Deterministic stand-ins for model providers.

These are teaching devices, not models. They make every example run offline
and give the same result each time, so the checks can show exactly what
changes between checkpoints.

ToyEmbedder
    Maps words to a small set of concepts so that different wording for the
    same idea lands close together ("forgot the PIN" ~ "PIN is
    unavailable"). That lexicon stands in for what a trained embedding model
    learns from data. It also ignores tokens containing digits, which
    exaggerates a real weakness of dense retrieval: exact identifiers such as
    E-417 carry little meaning for it.

ToyGenerator
    Doesn't write prose. It repeats the first sentence of each supplied source
    with its label, so you can see what evidence a real model would receive.
"""

from __future__ import annotations

import re

from atlas.providers import Vector

_STOPWORDS = set(
    "a an and are as at be before by do does for from get how i if in into is it its "
    "of on or the this to under used was we what when where which while with "
    "should my me our".split()
)

# Phrases are rewritten before tokenizing so multi-word expressions match.
_PHRASES = {
    "get back into": "regain access",
    "log in": "login",
    "how long": "duration",
}

# word -> concept. Words not listed stand for themselves.
_LEXICON = {
    **dict.fromkeys(["pin", "password", "passcode", "login"], "credential"),
    **dict.fromkeys(["lost", "losing", "lose", "unavailable", "missing", "forgot", "forgotten", "forgetting"], "unavailable"),
    **dict.fromkeys(["regain", "recover", "recovery", "recovering", "access", "unlock"], "access"),
    **dict.fromkeys(["warranty", "coverage", "covered", "guarantee"], "warranty"),
    **dict.fromkeys(["duration", "term", "years", "months"], "duration"),
    **dict.fromkeys(["battery", "batteries", "pack", "cell", "cells"], "battery"),
    **dict.fromkeys(["display", "screen", "dashboard"], "display"),
    **dict.fromkeys(["blink", "blinks", "flash", "flashes", "flashing", "blinking"], "blink"),
    **dict.fromkeys(["mudguard", "mudguards", "fender", "fenders"], "mudguard"),
    **dict.fromkeys(["check", "checks", "inspection", "inspect"], "check"),
    **dict.fromkeys(["requirement", "obligation", "rule"], "requirement"),
    **dict.fromkeys(["bike", "bicycle", "ebike", "c"], "bike"),
    **dict.fromkeys(["purchase", "purchased", "bought", "buy"], "purchase"),
    **dict.fromkeys(["capacity", "range"], "capacity"),
    **dict.fromkeys(["casing", "case", "housing"], "casing"),
}


def _stem(word: str) -> str:
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + replacement
    return word


def concepts(text: str) -> list[str]:
    text = text.lower()
    for phrase, replacement in _PHRASES.items():
        text = text.replace(phrase, replacement)
    result = []
    for word in re.findall(r"[a-z0-9]+", text):
        if word in _STOPWORDS or any(ch.isdigit() for ch in word):
            continue
        result.append(_LEXICON.get(word) or _LEXICON.get(_stem(word)) or _stem(word))
    return result


class ToyEmbedder:
    def embed(self, text: str) -> Vector:
        vector: Vector = {}
        for concept in concepts(text):
            vector[concept] = vector.get(concept, 0.0) + 1.0
        return vector


class ToyGenerator:
    def generate(self, prompt: str) -> str:
        lines = prompt.splitlines()
        quoted = []
        for i, line in enumerate(lines):
            if line.startswith("Source:") and i + 1 < len(lines):
                first_sentence = re.split(r"(?<=[.!?])\s+", lines[i + 1])[0]
                quoted.append(f"- {first_sentence} ({line.removeprefix('Source: ')})")
        if not quoted:
            return "(toy generator) No sources were supplied."
        return "(toy generator) The supplied sources say:\n" + "\n".join(quoted)
