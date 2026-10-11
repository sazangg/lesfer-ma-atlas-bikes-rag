"""Part 3: from parser output to structured documents you can trust.

A parser tells you what it found on each page. Before anything is indexed,
this step turns that into the document a reader would recognize, and checks
the result:

1. Drop page furniture (repeated headers and footers).
2. Apply corrections a person made to flagged text, and flag OCR text the
   parser was unsure about.
3. Rejoin tables split across pages, and attach the scope printed above them.
4. Group elements into sections (headings) or slides.
5. Describe figures with a vision model, so their content becomes text.
6. Report what was changed and what still needs a person.

The structured document keeps each element's type. Part 3 still renders it
to text for the size-based chunker; Part 4 uses the types directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from atlas.corpus import CORPUS_DIR, ParsedDocument
from atlas.providers import Describer

LOW_CONFIDENCE = 0.8


@dataclass
class Element:
    kind: str  # paragraph, table, figure, slide
    page: int | None
    data: dict

    def render(self) -> str:
        """Text for search and for the model. Part 4 refines what each kind supplies."""
        d = self.data
        if self.kind == "table":
            lines = [d["scope"]] if d.get("scope") else []
            lines.append("| " + " | ".join(d["header"]) + " |")
            lines.append("|" + " --- |" * len(d["header"]))
            lines += ["| " + " | ".join(row) + " |" for row in d["rows"]]
            return "\n".join(lines)
        if self.kind == "figure":
            return f"{d['caption']}\nFigure description: {d['description']}"
        if self.kind == "slide":
            notes = f"\nSpeaker notes: {d['notes']}" if d["notes"] else ""
            return f"{d['title']}\n{d['text']}{notes}"
        return d["text"]


@dataclass
class Section:
    number: str
    title: str
    meta: dict[str, str] = field(default_factory=dict)
    elements: list[Element] = field(default_factory=list)


@dataclass
class IngestReport:
    source_id: str
    file: str
    removed_furniture: int = 0
    merged_tables: int = 0
    corrected: list[str] = field(default_factory=list)
    needs_review: list[str] = field(default_factory=list)
    described_figures: int = 0

    def lines(self) -> list[str]:
        notes = []
        if self.removed_furniture:
            notes.append(f"removed {self.removed_furniture} page headers/footers")
        if self.merged_tables:
            notes.append(f"rejoined {self.merged_tables} table split across pages")
        if self.described_figures:
            notes.append(f"described {self.described_figures} figure(s)")
        notes += [f"applied reviewed correction: {c}" for c in self.corrected]
        notes += [f"NEEDS REVIEW: {n}" for n in self.needs_review]
        return notes or ["no changes"]


@dataclass
class StructuredDocument:
    source_id: str
    meta: dict[str, str]
    sections: list[Section]
    report: IngestReport


def load_corrections(directory: Path = CORPUS_DIR) -> dict[str, dict[str, str]]:
    path = directory / "corrections.json"
    return json.loads(path.read_text(encoding="utf-8"))["corrections"] if path.exists() else {}


def convert(doc: ParsedDocument, describer: Describer, corrections: dict[str, dict[str, str]]) -> StructuredDocument:
    report = IngestReport(doc.source_id, doc.file)
    fixes = corrections.get(doc.source_id, {})
    sections: list[Section] = []
    pending_scope = ""

    def section() -> Section:
        if not sections:
            sections.append(Section("", ""))
        return sections[-1]

    for raw in doc.elements:
        kind = raw["type"]

        # 1. Page furniture repeats on every page and carries no evidence.
        if kind in ("page_header", "page_footer"):
            report.removed_furniture += 1
            continue

        # 2. Reviewed corrections first; otherwise flag text the OCR was unsure about.
        text = raw.get("text")
        if raw["id"] in fixes:
            text = fixes[raw["id"]]
            report.corrected.append(f"{raw['id']} (page {raw.get('page')})")
        elif raw.get("ocr_confidence", 1.0) < LOW_CONFIDENCE:
            report.needs_review.append(
                f"{raw['id']} (page {raw.get('page')}) OCR confidence {raw['ocr_confidence']}: {text[:60]}..."
            )

        # 4. Headings and slides start sections.
        if kind == "heading":
            number, _, title = text.partition(" ")
            sections.append(Section(number, title, dict(raw.get("attrs", {}))))
            continue
        if kind == "slide":
            sections.append(Section(str(raw["slide"]), raw["title"]))
            section().elements.append(Element("slide", None, {
                "slide": raw["slide"], "title": raw["title"], "text": raw["text"], "notes": raw["notes"],
            }))
            continue

        current = section()
        if kind == "paragraph":
            if text.startswith("Scope:"):
                pending_scope = text
            current.elements.append(Element("paragraph", raw.get("page"), {"text": text}))
        elif kind == "table":
            previous = current.elements[-1] if current.elements else None
            # 3. A table without a header right after a table continues it.
            if raw["header"] is None and previous and previous.kind == "table":
                previous.data["rows"].extend(raw["rows"])
                report.merged_tables += 1
            else:
                current.elements.append(Element("table", raw.get("page"), {
                    "header": raw["header"], "rows": list(raw["rows"]), "scope": pending_scope,
                }))
            pending_scope = ""
        elif kind == "figure":
            # 5. The figure's values are in the image; a vision model turns them into text.
            description = describer.describe(raw["image"], raw["caption"])
            report.described_figures += 1
            current.elements.append(Element("figure", raw.get("page"), {
                "caption": raw["caption"], "image": raw["image"], "description": description,
            }))

    # 6. Verify what we can check mechanically: numbered rows should have no gaps.
    for sec in sections:
        for element in sec.elements:
            if element.kind == "table":
                gap = _numbering_gap([row[0] for row in element.data["rows"]])
                if gap:
                    report.needs_review.append(f"table in section {sec.number}: {gap}")
    return StructuredDocument(doc.source_id, doc.meta, sections, report)


def _numbering_gap(ids: list[str]) -> str | None:
    parsed = [re.fullmatch(r"([A-Z]+)(\d+)", i) for i in ids]
    if not all(parsed):
        return None
    numbers = [int(m.group(2)) for m in parsed]
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    return None if numbers == expected else f"row IDs not consecutive: {ids}"
