"""The assistant: prepare documents once, then answer questions.

Part 1 (naive RAG):

    Preparing:  parsed documents -> plain text -> size-based chunks -> embeddings -> vector index
    Answering:  question -> embedding -> top-k chunks -> prompt -> model -> answer

Part 3 replaces plain-text extraction with conversion and verification:

    Preparing:  parsed documents -> structured documents (+ ingestion report)
                -> rendered text -> size-based chunks -> ...
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atlas.chunking import Chunk, size_based_chunks
from atlas.convert import IngestReport, StructuredDocument, convert, load_corrections
from atlas.corpus import ParsedDocument, load_corpus
from atlas.extract import TextSection
from atlas.index import VectorIndex
from atlas.providers import Describer, Embedder, Generator
from atlas.toy import ToyDescriber, ToyEmbedder, ToyGenerator

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
        k: int = 5,
    ):
        self.embedder = embedder or ToyEmbedder()
        self.generator = generator or ToyGenerator()
        self.describer = describer or ToyDescriber()
        self.k = k
        self.index = VectorIndex()
        self.reports: list[IngestReport] = []

    # Preparing the documents ------------------------------------------------

    def ingest(self, documents: list[ParsedDocument], corrections: dict | None = None) -> None:
        corrections = load_corrections() if corrections is None else corrections
        for parsed in documents:
            doc = convert(parsed, self.describer, corrections)
            self.reports.append(doc.report)
            for chunk in size_based_chunks(doc.source_id, doc.meta, rendered_sections(doc)):
                self.index.add(chunk, self.embedder.embed(chunk.text))

    # Answering a question ---------------------------------------------------

    def ask(self, question: str, user: User = STAFF) -> Trace:
        trace = Trace(question)

        # Access is enforced before any passage can reach the prompt.
        hits = self.index.search(self.embedder.embed(question), user.groups, self.k)
        trace.retrieved = [(chunk.id, round(score, 3)) for score, chunk in hits]
        trace.context = [chunk for _, chunk in hits]

        trace.prompt = build_prompt(INSTRUCTIONS, question, trace.context)
        trace.answer = self.generator.generate(trace.prompt)
        return trace


def rendered_sections(doc: StructuredDocument) -> list[TextSection]:
    return [
        TextSection(s.number, s.title, "\n\n".join(e.render() for e in s.elements))
        for s in doc.sections
    ]


def build_prompt(instructions: str, question: str, context: list[Chunk]) -> str:
    sources = "\n\n".join(f"Source: {c.source_id}, section {c.section}\n{c.text}" for c in context)
    return f"{instructions}\n\nQuestion:\n{question}\n\n{sources}"


def default_assistant() -> Assistant:
    assistant = Assistant()
    assistant.ingest(load_corpus())
    return assistant
