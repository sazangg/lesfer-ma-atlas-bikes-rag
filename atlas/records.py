"""Part 9: an intermediate knowledge base of records.

Search returns the best few pieces. A question like "list every error code"
needs all of them, and no number of search results guarantees that. So at
ingestion, some content is also copied into records: one per error code
entry, one per battery safety check. Each record has fixed fields, keeps the
access groups of its document, and points back to its source section.

    record type   one record per                      fields
    -----------   ---------------------------------   -----------------------------
    error_code    error code entry (ERR-* documents)  code, product, component
    check         row of a battery safety check table  check_id, product, revision

A list question then filters records instead of searching passages: every
record that matches is returned, and the count is exact.

Records are only as complete as their extraction. `completeness_report`
compares them with the source structure, and `python -m atlas records`
writes them to knowledge/records.json so a change shows up in review like a
code change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from atlas.chunking import Chunk
from atlas.convert import StructuredDocument

RECORDS_FILE = Path(__file__).resolve().parent.parent / "knowledge" / "records.json"
CURATED_FILE = Path(__file__).resolve().parent.parent / "knowledge" / "curated.json"

RECORD_TYPES = {
    "error_code": "display error codes",
    "check": "battery safety checks",
}

# What a code is about, by the first of these words in its meaning. A real system
# asks a model to fill this field, and a person reviews it: list filters trust it.
COMPONENTS = {
    "battery": ["battery", "cell"],
    "charger": ["charger"],
    "motor": ["motor"],
    "light": ["light"],
    "sensor": ["sensor"],
}


def component_of(meaning: str) -> str | None:
    words = re.findall(r"[a-z]+", meaning.lower())
    for word in words:
        for component, terms in COMPONENTS.items():
            if word in terms:
                return component
    return None


def extract_records(doc: StructuredDocument) -> list[Chunk]:
    """Records found in one structured document. Each one is a Chunk, so access and
    situation filters work on it unchanged, and its ref is its source section."""
    records: list[Chunk] = []
    product = doc.meta.get("product", "")
    for section in doc.sections:
        base = {**doc.meta, **section.meta}

        code = section.meta.get("error_code")
        if code:
            text = " ".join(e.data["text"] for e in section.elements if e.kind == "paragraph")
            meaning = text.removeprefix(f"{code}:").strip()
            first_sentence = re.split(r"(?<=[.!?])\s+", meaning)[0]
            records.append(Chunk(
                f"{doc.source_id}-{section.number}-record", f"{code} ({product}): {meaning}",
                doc.source_id, section.number,
                {**base, "record_type": "error_code", "code": code, "component": component_of(first_sentence) or ""},
            ))

        if "check" in section.title.lower():
            for element in section.elements:
                if element.kind != "table":
                    continue
                for check_id, *cells in element.data["rows"]:
                    records.append(Chunk(
                        f"{doc.source_id}-{section.number}-{check_id}-record",
                        f"{check_id} ({product}): {' '.join(cells)}",
                        doc.source_id, section.number,
                        {**base, "record_type": "check", "check_id": check_id},
                    ))
    return records


def load_curated(path: Path = CURATED_FILE) -> list[Chunk]:
    """Records a person wrote, for knowledge the extracted records miss. Each cites its source."""
    if not path.exists():
        return []
    records = []
    for c in json.loads(path.read_text(encoding="utf-8"))["records"]:
        source_id, section = c["source"].split("#")
        meta = {k: v for k, v in c.items() if k not in ("id", "type", "source", "text", "note")}
        records.append(Chunk(c["id"], c["text"], source_id, section, {**meta, "record_type": c["type"], "curated": "yes"}))
    return records


def completeness_report(docs: list[StructuredDocument], records: list[Chunk]) -> list[str]:
    """Compare the records with what the documents contain, counted a second, independent way."""
    lines = []
    by_source: dict[str, list[Chunk]] = {}
    for r in records:
        by_source.setdefault(r.source_id, []).append(r)

    for doc in docs:
        found = by_source.get(doc.source_id, [])
        entries = [s for s in doc.sections if re.fullmatch(r"E-\d{3}", s.title)]
        if entries:
            got = sum(r.meta["record_type"] == "error_code" for r in found)
            lines.append(f"{doc.source_id}: {got} of {len(entries)} error code entries")
        rows = [row for s in doc.sections if "check" in s.title.lower()
                for e in s.elements if e.kind == "table" for row in e.data["rows"]]
        if rows:
            got = sum(r.meta["record_type"] == "check" for r in found)
            lines.append(f"{doc.source_id}: {got} of {len(rows)} check rows")
        for r in found:
            if r.meta["record_type"] == "error_code" and not r.meta.get("component"):
                lines.append(f"NEEDS REVIEW: {r.meta['code']} in {doc.source_id} has no component")

    # A code mentioned anywhere without an entry of its own means the list would be incomplete.
    known = {r.meta["code"] for r in records if r.meta["record_type"] == "error_code"}
    for doc in docs:
        for s in doc.sections:
            text = " ".join(e.render() for e in s.elements)
            for code in sorted(set(re.findall(r"\bE-\d{3}\b", text)) - known):
                lines.append(f"NEEDS REVIEW: {code} is mentioned in {doc.source_id} section {s.number} but has no entry")
    return lines


def save(records: list[Chunk], path: Path = RECORDS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"id": r.id, "type": r.meta["record_type"], "source": r.ref, "access": r.meta.get("access", ""),
             **{k: r.meta[k] for k in ("code", "component", "check_id", "product", "revision", "curated") if r.meta.get(k)},
             "text": r.text} for r in records]
    path.write_text(json.dumps({"records": rows}, indent=2) + "\n", encoding="utf-8")
