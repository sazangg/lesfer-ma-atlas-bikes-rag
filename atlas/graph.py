"""Part 11: when the evidence spans documents.

The battery recall names a part, BP-36-2, and no product. Which products use
that part is written elsewhere: in the C-3 specification, and in the S-2
warranty, which spells it "BP36-2". Search for the question finds the recall;
nothing in the question leads to the other two.

So at ingestion, every block is scanned for the things it mentions (here,
part numbers), and each mention becomes a link:

    block ──mentions──> part <──mentions── block

Answering then follows links from the best evidence to every other block that
mentions the same part, under the same access and situation filters.

Two blocks can only be linked if they name the part the same way, so names are
normalized first ("BP36-2" -> "BP-36-2"). That step is entity resolution, in
its simplest form.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from atlas.chunking import Chunk
from atlas.index import readable

PART_NUMBER = re.compile(r"\b([A-Z]{2})-?(\d{2})-(\d)\b")


def part_numbers(text: str) -> set[str]:
    """Part numbers in the catalog's form, however they were typed: BP36-2 and BP-36-2 are one part."""
    return {f"{family}-{size}-{version}" for family, size, version in PART_NUMBER.findall(text)}


class LinkGraph:
    def __init__(self) -> None:
        self.mentions: dict[str, list[Chunk]] = {}  # part -> blocks that mention it

    def add(self, block: Chunk) -> None:
        for part in part_numbers(block.text):
            self.mentions.setdefault(part, []).append(block)

    def follow(self, start: list[Chunk], groups: set[str], where: Callable[[Chunk], bool]) -> list[tuple[str, Chunk]]:
        """One hop: from the start blocks, through each part they mention, to the other blocks that mention it."""
        seen = {b.id for b in start}
        links = []
        for block in start:
            for part in sorted(part_numbers(block.text)):
                for other in self.mentions.get(part, []):
                    if other.id not in seen and readable(other, groups) and where(other):
                        seen.add(other.id)
                        links.append((part, other))
        return links
