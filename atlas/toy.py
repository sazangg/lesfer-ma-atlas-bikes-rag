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

ToyReranker
    Reads the question and a passage together and scores how much of the
    question the passage addresses. Identifiers count as question terms,
    weighted by how rare they are among the candidates. A real cross-encoder
    learns this judgment; the toy only shows where the step sits and what it
    returns.

ToyDescriber
    Reads a prepared description next to the image instead of looking at it.
    It stands in for a vision model's output. In a real pipeline a model
    writes that description and a person spot-checks the numbers.

ToyExtractor
    Finds a product name and a date with regular expressions. A real
    implementation asks a model for the same fields, with an instruction to
    return null rather than guess.

ToyRouter
    Recognizes a list question with a few rules: words like "list", "every",
    or "how many", and a record type it knows (error codes, battery checks).
    A real router asks a model to choose among the same paths, with a
    structured output. From part-10 it also recognizes a question asking for
    a share or rate, and the batches it names.

ToyPlanner
    Chooses the next tool call in the agent loop (part-12) with a few rules
    over the question and what earlier steps returned. A real planner is a
    model given the same tool descriptions, choosing by tool calling.

ToyGenerator
    Repeats a result computed by code first, word for word (part-10).
    Otherwise it doesn't write prose. It repeats the first sentence of each supplied source
    with its label, so you can see what evidence a real model would receive.
"""

from __future__ import annotations

import math
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


class ToyReranker:
    def score(self, question: str, passages: list[str]) -> list[float]:
        question_concepts = set(concepts(question))
        question_ids = set(re.findall(r"[a-z]+-[a-z]?\d+", question.lower()))
        passage_ids = [set(re.findall(r"[a-z]+-[a-z]?\d+", p.lower())) for p in passages]
        # Identifiers count as question terms. A rare one (an error code) counts
        # almost fully; one found in most candidates (a product name) barely counts.
        n = len(passages)
        id_weight = {
            i: math.log(n / max(1, sum(i in ids for ids in passage_ids))) / math.log(n) if n > 1 else 1.0
            for i in question_ids
        }
        terms = len(question_concepts) + len(question_ids) or 1
        scores = []
        for passage, ids in zip(passages, passage_ids):
            matched = len(question_concepts & set(concepts(passage)))
            matched += sum(id_weight[i] for i in question_ids & ids)
            # How much of the question does this passage address?
            scores.append(matched / terms)
        return scores


class ToyDescriber:
    def describe(self, image_path: str, caption: str) -> str:
        from atlas.corpus import CORPUS_DIR

        sidecar = (CORPUS_DIR / image_path).with_suffix(".description.txt")
        return sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else ""


_MONTHS = "january february march april may june july august september october november december".split()


class ToyExtractor:
    def extract(self, question: str) -> dict[str, str | None]:
        product = re.search(r"\b([A-Z]-\d+)\b", question)
        iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", question)
        spoken = re.search(r"\b(\d{1,2}) (" + "|".join(_MONTHS) + r") (\d{4})\b", question.lower())
        purchase_date = None
        if iso:
            purchase_date = iso.group(0)
        elif spoken:
            day, month, year = spoken.groups()
            purchase_date = f"{year}-{_MONTHS.index(month) + 1:02d}-{int(day):02d}"
        return {"product": product.group(1) if product else None, "purchase_date": purchase_date}


class ToyRouter:
    def route(self, question: str) -> dict[str, str | None]:
        q = question.lower()
        # part-13: what changed between two revisions of a document.
        revisions = re.search(r"\brevisions? (\d+) and (?:revision )?(\d+)\b", q)
        if revisions and re.search(r"\b(changed?|differ\w*|compare)\b", q):
            kind = next((k for k in ("battery warranty", "specification", "service manual", "assembly guide",
                                     "display error codes") if k in q), None)
            return {"path": "compare", "document": kind, "revisions": list(revisions.groups())}
        # part-12: a battery serial number means a path no fixed route knows in advance.
        if re.search(r"\b\d{4}-\d{4}\b", question):
            return {"path": "agent"}
        # part-11: "which bikes/products ..." asks for things the evidence may only reach through a link.
        if re.search(r"^which (bikes|models|products)\b", q):
            return {"path": "follow"}
        # part-10: a share or rate of something counted asks for a calculation, unless it asks what the rate means.
        if re.search(r"\b(share|percentage|pass rate)\b", q) and not re.search(r"\b(defined|definition|means?)\b", q):
            named = re.search(r"\bbatch(?:es)?\s+((?:[A-Z]\b(?:,\s*|\s+and\s+)?)+)", question)
            return {"path": "calculate", "metric": "pass_rate" if re.search(r"\b(pass|passed|inspection)\b", q) else None,
                    "batches": re.findall(r"[A-Z]", named.group(1)) if named else []}
        record_type ="error_code" if re.search(r"\bcodes?\b", q) else "check" if re.search(r"\bchecks?\b", q) else None
        # A list question asks for a set: "list", "every", "all", "how many", or "which" with a plural.
        wants_set = re.search(r"\b(list|every|all|how many)\b", q) or re.search(r"^which\b.*\b(codes|checks)\b", q)
        if not (record_type and wants_set):
            return {"path": "search", "record_type": None, "component": None}
        rest = re.sub(r"battery safety checks?", "", q)
        component = next((c for c in ("battery", "charger", "motor", "light", "sensor") if re.search(rf"\b{c}", rest)), None)
        return {"path": "list", "record_type": record_type, "component": component}


class ToyPlanner:
    """Decides each step from what the earlier steps returned, with a few rules a real model would replace."""

    def next_step(self, question: str, steps: list, tools: dict[str, str]) -> dict:
        from atlas.graph import part_numbers

        product = re.search(r"\b([A-Z]-\d+)\b", question)
        serial = re.search(r"\b\d{4}-\d{4}\b", question)
        if not (product and serial):
            missing = " and ".join(x for x, found in (("product", product), ("battery serial number", serial)) if not found)
            return {"thought": f"I can't start without the {missing}.", "tool": "finish",
                    "args": {"clarify": f"Which {missing} is this about?"}}
        product, serial = product.group(1), serial.group(0)
        done = {s.tool: s for s in steps}

        if "search" not in done:
            return {"thought": f"The serial alone doesn't say which pack it is. Find the pack the {product} uses.",
                    "tool": "search", "args": {"query": f"{product} battery pack"}}
        # Only a passage about this product says which pack it uses.
        mine = [b for b in done["search"].blocks if b.meta.get("product") == product and part_numbers(b.text)]
        if not mine:
            return {"thought": f"No passage says which pack the {product} uses.", "tool": "finish", "args": {"evidence": []}}
        part = sorted(part_numbers(mine[0].text))[0]
        evidence = [mine[0].ref]

        if "recall_status" not in done:
            return {"thought": f"The {product} uses {part}. Check serial {serial} against recalls of {part}.",
                    "tool": "recall_status", "args": {"part": part, "serial": serial}}
        check = done["recall_status"]
        evidence += [b.ref for b in check.blocks]
        if check.observation.startswith("affected: yes"):
            source = check.blocks[0].source_id
            if "read" not in done:
                return {"thought": f"It is affected. Read {source} for what to do.", "tool": "read",
                        "args": {"source_id": source}}
            evidence += [b.ref for b in done["read"].blocks]
            return {"thought": "I have the pack, the recall check, and the instructions.", "tool": "finish",
                    "args": {"evidence": list(dict.fromkeys(evidence))}}
        return {"thought": "It is not affected; nothing more to look up.", "tool": "finish",
                "args": {"evidence": list(dict.fromkeys(evidence))}}


class ToyGenerator:
    def generate(self, prompt: str) -> str:
        lines = prompt.splitlines()
        quoted = []
        for i, line in enumerate(lines):
            label = re.match(r"^\[(S\d+)\]", line)
            if label and i + 1 < len(lines):
                # Skip a section heading such as "2 Warranty term" to quote the first real sentence.
                heading = re.match(r"^\d[\d.]* \S", lines[i + 1]) and i + 2 < len(lines)
                body = lines[i + 2] if heading else lines[i + 1]
                first_sentence = re.split(r"(?<=[.!?])\s+", body)[0]
                quoted.append(f"- {first_sentence} [{label.group(1)}]")
        # part-10: a computed result comes first, word for word, as the instructions ask.
        prefix = "Result (computed by code): "
        result = next((line.removeprefix(prefix) for line in lines if line.startswith(prefix)), None)
        if result:
            quoted.insert(0, f"- {result}")
        # part-13: so do changes computed by code, one line per section.
        if "Changes (computed by code):" in lines:
            start = lines.index("Changes (computed by code):") + 1
            end = lines.index("", start) if "" in lines[start:] else len(lines)
            quoted[:0] = [f"- {line}" for line in lines[start:end]]
        if not quoted:
            return "(toy generator) No sources were supplied."
        return "(toy generator) The supplied sources say:\n" + "\n".join(quoted)
