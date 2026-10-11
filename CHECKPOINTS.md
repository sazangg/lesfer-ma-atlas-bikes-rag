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

## part-02: Name the failures before fixing them

**Article:** When a useful answer isn't a correct answer

**Lands in:** around the pipeline. The pipeline itself doesn't change.

- `cases/cases.json` records real questions with their expected behavior and the part meant to handle each one.
- `python -m atlas check` runs them and labels each result PASS, KNOWN GAP, or REGRESSION.
- `--save-traces` writes the question, retrieved chunks, prompt, and answer for each case to `traces/`, so you can see where a failure happened.

This is the lightweight start of evaluation. Part 7 builds it into proper measurement.

**Answers:** the same questions as part-01. What changes is that you can see which questions fail, and which part is meant to fix each:

```
KNOWN GAP -> part-03   Q08-scanned-exception    'the rear mudguard is not required' not supplied
KNOWN GAP -> part-03   Q09-slide-notes          'speed sensor signal is lost' not supplied
KNOWN GAP -> part-04   Q06-mudguard-exception   not in the same passage
KNOWN GAP -> part-05   Q02-error-code           missing ['ERR-C3-R2#4.17']
KNOWN GAP -> part-06   Q01-applicable-only      also supplied ['WARR-R1#2', 'WARR-S2-R1#2']
KNOWN GAP -> part-09   Q04-every-check          missing ['R03', 'R04', 'R05']; completeness not guaranteed
```

**Not a stopping point.** Keep this case file and runner whatever you build next. Every later checkpoint is judged by it.

## part-03: From raw files to structured documents

**Article:** From raw files to structured documents

**Lands in:** ingestion. Answering doesn't change.

```
Preparing:  parsed documents
            -> convert: drop page furniture, apply reviewed corrections, flag unsure OCR,
                        rejoin split tables with their scope, keep slide notes,
                        describe figures with a vision model
            -> structured documents + ingestion report
            -> rendered text -> size-based chunks -> index
```

In many real projects this layer is most of the work. `convert.py` keeps each element's type (paragraph, table, figure, slide). Here it is still rendered to text for the part-01 chunker; part-04 uses the types.

`python -m atlas ingest-report --no-corrections` shows what a person must check before the system is trusted:

```
ASSEMBLY-C3-R3 (c3-assembly-guide-r3-scan.pdf)
  - NEEDS REVIEW: e5 (page 9) OCR confidence 0.61: Exception: on the cargo version of the C-3, the rear mudguar...
SPEC-C3-R5 (c3-specification-r5.pdf)
  - removed 6 page headers/footers
  - rejoined 1 table split across pages
TEST-BAT-R1 (battery-cycle-test-r1.pdf)
  - removed 2 page headers/footers
  - described 1 figure(s)
```

The reviewer's fix lives in `corpus/corrections.json`, keyed by the parser's element ID, so it survives re-ingestion.

**Closed gaps:**
- `Q08-scanned-exception`: the cargo exception reaches the model as "the rear mudguard is not required", not the OCR's "requlred".
- `Q09-slide-notes`: the meaning of two blinks, which exists only in speaker notes, reaches the model.
- `Q11-figure-values`: "82 percent", printed only in the figure, reaches the model through its description.

**Still can't:**
- Keep the mudguard rule with its cargo exception, or check R07 with its scope: size-based chunks still cut through them.
- Give the model the figure itself, not only a description of it.
- Find the `E-417` entry.

**Stop here if** your documents are mostly prose with simple layout, and your questions don't depend on tables, exceptions, or figures staying intact.

## part-04: From documents to evidence

**Article:** From documents to evidence

**Lands in:** ingestion, plus one answering step.

What is easy to find is often not what the model needs to read. Each kind of content now gets a **searched form** and a **supplied form** (`evidence.py`):

| Kind | Searched as | Supplied to the model as |
| --- | --- | --- |
| paragraph | the paragraph, with its section title | the whole section |
| table row | the row, with its scope and section | the whole table, header and scope included |
| figure | caption + description | caption, description, image file, page |
| slide | title, text, speaker notes | the whole slide |

```
Preparing:  structured documents -> searchable pieces (indexed) + blocks (stored)
Answering:  question -> top-5 pieces -> the block each belongs to, once -> prompt
```

The size-based chunker from part-01 stays in `chunking.py` for comparison; the pipeline no longer uses it.

**Closed gaps:**
- `Q06-mudguard-exception`: the mudguard rule arrives with its cargo exception, because the model receives section 4.2 whole.
- `Q07-check-scope`: check R07 arrives inside its table, with "C-3, 36 V battery pack, frame-mounted, standard charger".
- `Q12-figure-itself`: the model receives the figure (image file and page), so a model that reads images can check the description against the image, and a person can open page 8.

**Worth noticing:** `Q04-every-check` now reports `all present; completeness not guaranteed`. The table is small, so supplying it whole happens to include all eleven checks. Nothing guarantees that for a larger register or for checks spread over several documents. That difference is Part 9's subject.

**Still can't:** find the `E-417` entry. Asked "What does error E-417 mean, and what should I check?", the top result is the battery safety checks table: it matches "check", and the toy embedder can't see the code.

**Stop here if** your questions are answered by finding one passage, table, figure, or slide, and plain wording is enough to find it.

## part-05: When search misses the evidence

**Article:** When search misses the evidence

**Lands in:** retrieval. Ingestion now also builds a keyword index next to the vector index.

```
                       exact code lookup ─────────────────────────┐
question ─> keyword search (BM25) ┐                                ├─> top-5 pieces -> blocks -> prompt
         └> vector search ────────┴─> fusion (RRF) -> reranker ───┘
```

- `KeywordIndex` (BM25) keeps identifiers such as `E-417` as exact terms.
- An error code in the question resolves directly to its entry. The `error_code` field comes from the HTML heading attributes that part-03's conversion preserved; part-01's plain-text extraction threw them away.
- Reciprocal rank fusion combines the two searches by position, so their incompatible scores never need to be added.
- A reranker reads the question and each shortlisted piece together. Two limits now exist: candidates per search (10) and pieces kept (5).
- Non-text content needs nothing special at this stage: table rows, figure descriptions, and slides are searched through their searched form, and part-04 still supplies the whole block. "Which check covers the battery vent holes?" hits row R08 first, and the model receives the whole table.

**Closed gaps:** `Q02-error-code` and `Q02-exact-code-first`. The `E-417` entry is now first:

```
    inf  ERR-C3-R2-4.17-p0
  0.400  TRAIN-C3-2-search
```

**Two regressions caught while building this checkpoint.** The first reranker scored word overlap divided by passage length, plus a large bonus for every shared identifier. The checker reported:

```
REGRESSION             Q09-slide-notes          'speed sensor signal is lost' not supplied
```

Both searches had ranked the right slide first. The reranker demoted it below passages that merely contained "C-3", which the question mentions and the slide doesn't. A second attempt scored how much of the question a passage covers, plus a bonus for identifiers that are rare among the candidates. It failed the same case for the same reason: the bonus still outweighed the content. The version that shipped treats identifiers as ordinary question terms, weighted by rarity, inside the same coverage score. A passage now wins by addressing more of the question. A reranker is a model making judgments; without the case file, either earlier version would have shipped.

**Still can't:**
- Choose between warranty policies by purchase date: both C-3 policies still reach the model.
- List every safety check with a guarantee.

**Stop here if** your users ask single-answer questions, some with exact codes or names and some in their own words, and your documents don't come in versions that apply to different situations.
