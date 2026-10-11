"""Part 1: get plain text out of each document, the quick way.

This is what most first prototypes do: take whatever text the parser found,
in order, and move on. It keeps page headers and footers, turns table rows
into loose lines, keeps only a figure's caption, drops slide notes, and
passes OCR mistakes straight through. Part 3 replaces it.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.corpus import ParsedDocument


@dataclass
class TextSection:
    number: str
    title: str
    text: str


def plain_text_sections(doc: ParsedDocument) -> list[TextSection]:
    sections: list[TextSection] = []

    def current() -> TextSection:
        if not sections:
            sections.append(TextSection("", "", ""))
        return sections[-1]

    def add(text: str) -> None:
        section = current()
        section.text = f"{section.text}\n{text}" if section.text else text

    for element in doc.elements:
        kind = element["type"]
        if kind == "heading":
            number, _, title = element["text"].partition(" ")
            sections.append(TextSection(number, title, ""))
        elif kind == "slide":
            sections.append(TextSection(str(element["slide"]), element["title"], ""))
            add(f"{element['title']}\n{element['text']}")
        elif kind == "table":
            for row in ([element["header"]] if element["header"] else []) + element["rows"]:
                add(" ".join(row))
        elif kind == "figure":
            add(element["caption"])
        else:  # paragraph, page_header, page_footer
            add(element["text"])
    return sections
