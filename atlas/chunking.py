"""Turn text into searchable chunks.

Part 1 uses the simplest reasonable rule: pack text into pieces of at most
`max_chars` characters, splitting long paragraphs into lines and long lines
into sentences. It ignores what the text means.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from atlas.extract import TextSection


@dataclass
class Chunk:
    id: str
    text: str
    source_id: str
    section: str
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.source_id}#{self.section}"


def _pieces(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    for line in paragraph.splitlines():
        if len(line) <= max_chars:
            pieces.append(line)
        else:
            pieces.extend(re.split(r"(?<=[.!?])\s+", line))
    return pieces


def size_based_chunks(
    source_id: str, meta: dict[str, str], sections: list[TextSection], max_chars: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(section: TextSection, text: str) -> None:
        chunks.append(Chunk(f"{source_id}-{section.number}-chunk-{len(chunks)}", text, source_id, section.number, dict(meta)))

    for section in sections:
        current = ""
        for piece in _pieces(section.text, max_chars):
            if current and len(current) + 1 + len(piece) > max_chars:
                emit(section, current)
                current = piece
            else:
                current = f"{current}\n{piece}" if current else piece
        if current:
            emit(section, current)
    return chunks
