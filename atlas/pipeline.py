"""The assistant: prepare documents once, then answer questions.

Part 1 (naive RAG):

    Preparing:  parsed documents -> plain text -> size-based chunks -> embeddings -> vector index
    Answering:  question -> embedding -> top-k chunks -> prompt -> model -> answer

Part 3 replaces plain-text extraction with conversion and verification:

    Preparing:  parsed documents -> structured documents (+ ingestion report) -> ...

Part 4 gives each kind of content a searched form and a supplied form:

    Preparing:  structured documents -> searchable pieces (indexed) + blocks (stored)
    Answering:  ... top-k pieces -> the blocks they belong to, once each -> prompt ...

Part 5 replaces the single vector search with a retrieval stage:

    exact code lookup ┐
    keyword (BM25)    ├─> reciprocal rank fusion -> rerank -> top-k
    vector search     ┘
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from atlas.chunking import Chunk
from atlas.convert import IngestReport, convert, load_corrections
from atlas.corpus import ParsedDocument, load_corpus
from atlas.evidence import evidence_units
from atlas.index import VectorIndex
from atlas.keyword import KeywordIndex, reciprocal_rank_fusion
from atlas.providers import Describer, Embedder, Generator, Reranker
from atlas.toy import ToyDescriber, ToyEmbedder, ToyGenerator, ToyReranker

INSTRUCTIONS = """Answer the question using the supplied passages.
Identify the source of your answer.
If the passages don't establish an answer, say so."""


@dataclass
class User:
    name: str
    groups: set[str]


STAFF = User("service staff", {"employees", "staff"})


@dataclass
class Trace:
    """Everything that happened for one question. Later parts check these fields."""

    question: str
    retrieved: list[tuple[str, float]] = field(default_factory=list)
    context: list[Chunk] = field(default_factory=list)
    prompt: str = ""
    answer: str = ""
    status: str = "answered"


class Assistant:
    def __init__(
        self,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        describer: Describer | None = None,
        reranker: Reranker | None = None,
        k: int = 5,
        candidates: int = 10,
    ):
        self.embedder = embedder or ToyEmbedder()
        self.generator = generator or ToyGenerator()
        self.describer = describer or ToyDescriber()
        self.reranker = reranker or ToyReranker()
        self.k = k  # pieces supplied to the model (before expanding to blocks)
        self.candidates = candidates  # pieces each search contributes to the shortlist
        self.index = VectorIndex()
        self.keywords = KeywordIndex()
        self.blocks: dict[str, Chunk] = {}
        self.reports: list[IngestReport] = []

    # Preparing the documents ------------------------------------------------

    def ingest(self, documents: list[ParsedDocument], corrections: dict | None = None) -> None:
        corrections = load_corrections() if corrections is None else corrections
        for parsed in documents:
            doc = convert(parsed, self.describer, corrections)
            self.reports.append(doc.report)
            searchable, blocks = evidence_units(doc)
            self.blocks.update((b.id, b) for b in blocks)
            for chunk in searchable:
                text = chunk.search_text or chunk.text
                self.index.add(chunk, self.embedder.embed(text))
                self.keywords.add(chunk, text)

    # Answering a question ---------------------------------------------------

    def ask(self, question: str, user: User = STAFF) -> Trace:
        trace = Trace(question)

        hits = self.retrieve(question, user)
        trace.retrieved = [(chunk.id, round(score, 3)) for score, chunk in hits]
        trace.context = self.supplied_blocks([chunk for _, chunk in hits])

        trace.prompt = build_prompt(INSTRUCTIONS, question, trace.context)
        trace.answer = self.generator.generate(trace.prompt)
        return trace

    def retrieve(self, question: str, user: User) -> list[tuple[float, Chunk]]:
        """Return the top-k pieces for the question. Access is enforced by every search."""
        groups = user.groups

        # An exact identifier in the question resolves directly to its entry.
        codes = set(re.findall(r"\b[A-Z]+-[A-Z]?\d+\b", question))
        exact = self.index.lookup("error_code", codes, groups)

        lexical = [c for _, c in self.keywords.search(question, groups, self.candidates)]
        dense = [c for _, c in self.index.search(self.embedder.embed(question), groups, self.candidates)]
        shortlist = [c for c in reciprocal_rank_fusion(lexical, dense) if c not in exact]

        scores = self.reranker.score(question, [c.search_text or c.text for c in shortlist])
        ranked = sorted(zip(scores, shortlist), key=lambda pair: pair[0], reverse=True)
        return ([(float("inf"), c) for c in exact] + ranked)[: self.k]

    def supplied_blocks(self, hits: list[Chunk]) -> list[Chunk]:
        """Replace each searched piece by the block it belongs to, once."""
        context: list[Chunk] = []
        seen: set[str] = set()
        for chunk in hits:
            block = self.blocks[chunk.parent_id] if chunk.parent_id else chunk
            if block.id not in seen:
                seen.add(block.id)
                context.append(block)
        return context


def build_prompt(instructions: str, question: str, context: list[Chunk]) -> str:
    sources = "\n\n".join(f"Source: {c.source_id}, section {c.section}\n{c.text}" for c in context)
    return f"{instructions}\n\nQuestion:\n{question}\n\n{sources}"


def default_assistant() -> Assistant:
    assistant = Assistant()
    assistant.ingest(load_corpus())
    return assistant
