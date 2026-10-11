"""Interfaces for the model-backed parts of the system.

The pipeline only depends on these protocols. `atlas.toy` provides
deterministic stand-ins so everything runs offline; swap in a real
embedding model or LLM by implementing the same methods.
"""

from __future__ import annotations

from typing import Protocol

# A vector is a mapping from dimension to weight. Real embedding models return
# dense arrays of floats; a sparse mapping keeps the toy version readable.
Vector = dict[str, float]


class Embedder(Protocol):
    def embed(self, text: str) -> Vector: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


def cosine(a: Vector, b: Vector) -> float:
    dot = sum(weight * b.get(dim, 0.0) for dim, weight in a.items())
    norm_a = sum(w * w for w in a.values()) ** 0.5
    norm_b = sum(w * w for w in b.values()) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
