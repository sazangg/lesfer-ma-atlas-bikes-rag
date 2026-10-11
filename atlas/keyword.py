"""Keyword search with BM25, plus the pieces that combine search results.

BM25 scores a chunk by the question's terms it contains. Rare terms count for
more, repeating a term has diminishing returns, and long chunks are
normalized so they don't win just by being long. Unlike the toy embedder, it
keeps identifiers such as E-417 as exact terms.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from atlas.chunking import Chunk
from atlas.index import _readable

_STOPWORDS = set("a an and are as at be by do does for how i in is it of on or the this to what when with".split())


def terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [_plural(t) for t in tokens if t not in _STOPWORDS]


def _plural(token: str) -> str:
    return token[:-1] if token.endswith("s") and len(token) > 4 and not any(c.isdigit() for c in token) else token


class KeywordIndex:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._entries: list[tuple[Chunk, Counter]] = []

    def add(self, chunk: Chunk, text: str) -> None:
        self._entries.append((chunk, Counter(terms(text))))

    def search(
        self, question: str, allowed_groups: set[str], limit: int, where: Callable[[Chunk], bool] | None = None
    ) -> list[tuple[float, Chunk]]:
        n = len(self._entries)
        avg_len = sum(sum(c.values()) for _, c in self._entries) / n
        doc_freq = Counter(t for _, counts in self._entries for t in counts)
        scored = []
        for chunk, counts in self._entries:
            if not _readable(chunk, allowed_groups) or (where is not None and not where(chunk)):
                continue
            length = sum(counts.values())
            score = 0.0
            for term in set(terms(question)):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / avg_len))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]


def reciprocal_rank_fusion(*rankings: list[Chunk], c: int = 60) -> list[Chunk]:
    """Combine ranked lists by position: each list adds 1 / (c + rank)."""
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (c + rank)
            by_id[chunk.id] = chunk
    return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]
