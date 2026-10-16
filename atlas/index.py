"""An in-memory vector index.

A vector database does the same job at scale: store vectors with their
records and return the nearest ones. Nearest doesn't mean relevant; the index
always returns something.
"""

from __future__ import annotations

from collections.abc import Callable

from atlas.chunking import Chunk
from atlas.providers import Vector, cosine


class VectorIndex:
    def __init__(self) -> None:
        self._entries: list[tuple[Chunk, Vector]] = []

    def add(self, chunk: Chunk, vector: Vector) -> None:
        self._entries.append((chunk, vector))

    def search(
        self, vector: Vector, allowed_groups: set[str], limit: int, where: Callable[[Chunk], bool] | None = None
    ) -> list[tuple[float, Chunk]]:
        scored = [
            (cosine(vector, chunk_vector), chunk)
            for chunk, chunk_vector in self._entries
            if readable(chunk, allowed_groups) and (where is None or where(chunk))
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]

    def lookup(
        self, field: str, values: set[str], allowed_groups: set[str], where: Callable[[Chunk], bool] | None = None
    ) -> list[Chunk]:
        """Exact match on a metadata field, such as an error code."""
        return [
            chunk
            for chunk, _ in self._entries
            if chunk.meta.get(field) in values and readable(chunk, allowed_groups) and (where is None or where(chunk))
        ]

    def __len__(self) -> int:
        return len(self._entries)


def readable(chunk: Chunk, groups: set[str]) -> bool:
    access = {g.strip() for g in chunk.meta.get("access", "").split(",") if g.strip()}
    return bool(access & groups)
