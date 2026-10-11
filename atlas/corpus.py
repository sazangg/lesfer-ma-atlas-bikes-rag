"""Load the Atlas corpus: parser output for each source document.

Each file in corpus/parsed/ is what a layout-aware parser returns for one
PDF, slide deck, web page, or Word document: typed elements with page
numbers. See corpus/README.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


@dataclass
class ParsedDocument:
    source_id: str
    file: str
    format: str
    meta: dict[str, str]
    elements: list[dict]


def load_corpus(directory: Path = CORPUS_DIR) -> list[ParsedDocument]:
    documents = []
    for path in sorted((directory / "parsed").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        documents.append(ParsedDocument(data["source_id"], data["file"], data["format"], data["meta"], data["elements"]))
    return documents
