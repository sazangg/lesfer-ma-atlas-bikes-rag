# Atlas Bikes: a knowledge assistant, one gap at a time

Reference code for the series *Building Reliable AI Over Company Knowledge*.

Atlas Bikes is a fictional e-bike company. This repository builds an assistant over its documents, starting from naive RAG. Each article in the series closes one gap with one change, and each change is a tagged checkpoint here.

**The system evolves to close a gap. It is not a destination.** Every checkpoint is a complete system for the questions it answers. If your users' questions stop at Part 5, Part 5 is your system. See [CHECKPOINTS.md](CHECKPOINTS.md) for what each one answers and what it still can't.

## Run it

Python 3.11+, no dependencies.

```bash
python -m atlas ask "What battery warranty applies to a C-3 bought on 15 March 2026?"
python -m atlas ask "What does error E-417 mean?" --show-prompt
```

Move between checkpoints with git:

```bash
git checkout part-03
git diff part-02 part-03
```

## Offline by default

The documents in `corpus/parsed/` are the output of a document parser for PDFs, a scanned assembly guide, a training slide deck, a web page, and Word files. They ship instead of the originals so nothing needs installing. See [corpus/README.md](corpus/README.md).

The model-backed parts sit behind small interfaces in `atlas/providers.py`. The defaults in `atlas/toy.py` are deterministic stand-ins, so every example runs without an API key and gives the same result each time:

- The **toy embedder** maps words to a tiny set of concepts. It stands in for what a trained embedding model learns, and it deliberately ignores tokens containing digits, exaggerating a real weakness of dense retrieval with identifiers like `E-417`.
- The **toy generator** doesn't write prose. It repeats the first sentence of each supplied source with its label, so you can see exactly what a real model would receive.

Implement the same methods to plug in a real parser, embedding model, or LLM. The checks inspect evidence, not prose, so they stay meaningful with the toys.

## Layout

```
corpus/        parser output for the Atlas documents, plus figures
atlas/         the assistant
```
