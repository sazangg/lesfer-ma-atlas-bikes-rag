"""Part 4: from structured documents to evidence.

What is easy to find is often not what the model needs to read. So every
piece of content gets two forms:

    kind        searched as                            supplied to the model as
    ---------   ------------------------------------   ------------------------------------------
    paragraph   the paragraph, with its section title  the whole section (rule + its exception)
    table row   the row, with its scope and section    the whole table, header and scope included
    figure      caption + description                  caption, description, image file, page
    slide       title, text, speaker notes             the whole slide

Search returns small pieces. Each piece points to the block it belongs to
(`parent_id`), and the model receives that block.
"""

from __future__ import annotations

from atlas.chunking import Chunk
from atlas.convert import StructuredDocument


def evidence_units(doc: StructuredDocument) -> tuple[list[Chunk], list[Chunk]]:
    """Return (searchable pieces, blocks supplied to the model)."""
    searchable: list[Chunk] = []
    blocks: list[Chunk] = []

    for section in doc.sections:
        meta = {**doc.meta, **section.meta}
        heading = f"{section.number} {section.title}".strip()
        # part-07: the document title is searched too ("Dealer bulletin 2026-02: charger recall").
        title = doc.meta.get("title", "")

        def block(suffix: str, text: str, kind: str) -> Chunk:
            b = Chunk(f"{doc.source_id}-{section.number}{suffix}", text, doc.source_id, section.number,
                      {**meta, "element": kind})
            blocks.append(b)
            return b

        prose = [e for e in section.elements if e.kind == "paragraph" and not e.data["text"].startswith("Scope:")]
        if prose:
            parent = block("", "\n".join([heading] + [e.render() for e in prose]), "section")
            for n, element in enumerate(prose):
                searchable.append(Chunk(
                    f"{parent.id}-p{n}", element.render(), doc.source_id, section.number, meta,
                    parent_id=parent.id, search_text=f"{title}\n{section.title}\n{element.render()}",
                ))

        for n, element in enumerate(e for e in section.elements if e.kind != "paragraph"):
            d = element.data
            if element.kind == "table":
                table = block(f"-table{n}", f"{heading}\n{element.render()}", "table")
                for record_id, *cells in d["rows"]:
                    row = f"{record_id}: {' '.join(cells)}"
                    searchable.append(Chunk(
                        f"{table.id}-{record_id}", row, doc.source_id, section.number,
                        {**meta, "record_id": record_id},
                        parent_id=table.id, search_text=f"{title}\n{section.title}\n{d['scope']}\n{row}",
                    ))
            elif element.kind == "figure":
                figure = block(f"-figure{n}", (
                    f"{d['caption']}\nFigure description: {d['description']}\n"
                    f"Image: {d['image']} (page {element.page})"
                ), "figure")
                searchable.append(Chunk(
                    f"{figure.id}-search", d["caption"], doc.source_id, section.number, meta,
                    parent_id=figure.id, search_text=f"{title}\n{d['caption']}\n{d['description']}",
                ))
            elif element.kind == "slide":
                slide = block("", f"Slide {d['slide']}: {element.render()}", "slide")
                searchable.append(Chunk(
                    f"{slide.id}-search", d["title"], doc.source_id, section.number, meta,
                    parent_id=slide.id, search_text=f"{title}\n{element.render()}",
                ))
    return searchable, blocks
