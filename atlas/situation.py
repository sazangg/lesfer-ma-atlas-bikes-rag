"""Part 5: which documents can apply to the question's situation?

Some documents only apply to a product or a period. The warranty policies
carry `product`, `applies_from`, and `applies_through` in their metadata.
A model extracts the matching facts from the question; ordinary code then
filters the search so inapplicable chunks are never candidates.

This is metadata filtering. It works for any effective-dated or
product-specific document, not only warranties.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from atlas.chunking import Chunk


@dataclass
class Situation:
    product: str | None = None
    on_date: date | None = None

    def allows(self, chunk: Chunk) -> bool:
        product = chunk.meta.get("product")
        if self.product and product and product != self.product:
            return False
        start, end = chunk.meta.get("applies_from"), chunk.meta.get("applies_through")
        if self.on_date and start and self.on_date < date.fromisoformat(start):
            return False
        if self.on_date and end and self.on_date > date.fromisoformat(end):
            return False
        return True


def is_date_scoped(chunk: Chunk) -> bool:
    return "applies_from" in chunk.meta


def missing_details(situation: Situation, hits: list[Chunk]) -> list[str]:
    """Details we must ask for before date-scoped evidence can be used."""
    if not hits or not is_date_scoped(hits[0]):
        return []
    missing = []
    products = {c.meta.get("product") for c in hits if is_date_scoped(c)}
    if situation.product is None and len(products) > 1:
        missing.append("product")
    if situation.on_date is None:
        missing.append("purchase date")
    return missing
