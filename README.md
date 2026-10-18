# Atlas Bikes: a knowledge assistant, one gap at a time

Reference code for the series *Building Reliable AI Over Company Knowledge*.

Atlas Bikes is a fictional e-bike company. This repository builds an assistant over its documents, starting from naive RAG. Each article in the series closes one gap with one change, and each change is a tagged checkpoint here.

**The system evolves to close a gap. It is not a destination.** Every checkpoint is a complete system for the questions it answers. If your users' questions stop at Part 5, Part 5 is your system. See [CHECKPOINTS.md](CHECKPOINTS.md) for what each one answers and what it still can't.

## Following the series

Each article has a tag. Check it out to see the system exactly as that article leaves it.

| Part | Article | Tag |
| --- | --- | --- |
| 1 | How an AI assistant answers from company documents | `part-01` |
| 2 | When a useful answer isn't a correct answer | `part-02` |
| 3 | From raw files to structured documents | `part-03` |
| 4 | From documents to evidence | `part-04` |
| 5 | When search misses the evidence | `part-05` |
| 6 | Choosing the evidence that applies | `part-06` |
| 7 | Measuring whether it works | `part-07` |
| 8 | Watching it in production | `part-08` |
| 9 | When the question needs every match | `part-09` |
| 10 | When the answer must be calculated | `part-10` |
| 11 | When the answer spans documents | `part-11` |
| 12 | When the next step depends on what you find | `part-12` |
| 13 | Modules: whole documents | coming |
| 14 | Choosing your system | coming |

Parts 1 to 6 are the core every system needs. Parts 9 to 13 are independent modules: add only the ones your questions need.

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
cases/         questions with their expected behavior
corpus/        parser output for the Atlas documents, plus figures
atlas/         the assistant
tests/         unit tests (python -m unittest)
```

## Check a checkpoint

```bash
python -m atlas check
python -m atlas measure            # from part-07: retrieval metrics, results by kind, stability
python -m atlas judge-calibration  # from part-07: compare a judge with human labels
python -m atlas simulate           # from part-08: replay a week of traffic with logging on
python -m atlas stats              # from part-08: what happened, and the review queue
python -m atlas trace r007         # from part-08: one request, stage by stage
python -m atlas records            # from part-09: build the records and check them against the documents
```

Each case states the part meant to handle it. Failing cases that a later part will close are reported as known gaps, not errors; a failure the current checkpoint should handle is a regression.

## License

MIT, for the code and the Atlas Bikes documents alike. See [LICENSE](LICENSE). Reuse it freely.
