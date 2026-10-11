"""An in-memory vector index.

A vector database does the same job at scale: store vectors with their
records and return the nearest ones. Nearest doesn't mean relevant; the index
always returns something.
"""

from __future__ import annotations

from atlas.chunking import Chunk
from atlas.providers import Vector, cosine


class VectorIndex:
    def __init__(self) -> None:
        self._entries: list[tuple[Chunk, Vector]] = []

    def add(self, chunk: Chunk, vector: Vector) -> None:
        self._entries.append((chunk, vector))

    def search(self, vector: Vector, allowed_groups: set[str], limit: int) -> list[tuple[float, Chunk]]:
        scored = [
            (cosine(vector, chunk_vector), chunk)
            for chunk, chunk_vector in self._entries
            if _readable(chunk, allowed_groups)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]

    def __len__(self) -> int:
        return len(self._entries)


def _readable(chunk: Chunk, groups: set[str]) -> bool:
    access = {g.strip() for g in chunk.meta.get("access", "").split(",") if g.strip()}
    return bool(access & groups)
