# Checkpoints

Each entry says where the change lands, what the system now answers, what it still can't, and when you can stop.

| Section | Checkpoints |
| --- | --- |
| Core | part-01 to part-06 |
| Measure | part-07 to part-08 |
| Modules (pick what you need) | part-09 to part-13 |
| Closing | part-14 |

## part-01: Naive RAG

**Article:** How an AI assistant answers from company documents

**Lands in:** the whole pipeline, built for the first time.

```
Preparing:  parsed documents -> plain text -> size-based chunks -> toy embeddings -> vector index
Answering:  question -> embedding -> top-5 chunks (access-filtered) -> prompt -> model
```

Text comes out of the parser output the quick way (`extract.py`): everything in order, as plain text.

**Answers:** single-fact questions whose answer sits in one clean passage, such as the battery warranty for a C-3 bought in 2026, or how to get back into the display after forgetting the PIN.

**Still can't:**
- Give clean text from the scanned assembly guide: OCR mistakes ("requlred") reach the model.
- Answer from a slide's speaker notes or a figure's values: plain-text extraction drops both.
- Find the entry for display error `E-417` at all: vector search can't see codes, and the entry shares no other words with the question.
- Keep the old and new warranty policies apart, or list all eleven battery safety checks.

**Stop here if** your documents are clean digital text, your questions are single facts, and a person reads each answer with its sources.
