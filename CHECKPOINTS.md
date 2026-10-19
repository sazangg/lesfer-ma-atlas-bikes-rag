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

## part-06: Choosing the evidence that applies

**Article:** Choosing the evidence that applies

**Lands in:** retrieval and context assembly. No new path: every question still flows through the same pipeline.

```
question -> extract product + purchase date (model proposes, code validates)
         -> retrieval, every search filtered to pieces that apply to that situation
         -> best evidence is date-scoped but no date? -> ask for it, stop
         -> blocks labeled [S1], [S2] ... -> model
         -> code checks each label and turns it into a document link
```

- `situation.py`: a piece whose document has `product`, `applies_from`, or `applies_through` metadata is only a candidate when it matches the question's situation. That metadata comes from the document management system and has been in the parsed documents since part-01; nothing used it. This is **metadata filtering**, and it works for any document that applies to a product or a period.
- Code validates what the model extracts. A product must be one Atlas Bikes sells: in "What does error E-417 mean?", the toy extractor proposes `E-417` as a product, and validation discards it.
- Clarification is decided by the evidence, not by a warranty rule: if the best match only applies within a date range and the question has no date, the assistant asks. If policies for several products match, it asks for the product.
- Citations: the prompt labels blocks `[S1]`, the model cites labels, and `resolve_citations` maps them to links. An unknown label withholds the answer. A valid label doesn't prove the block supports the claim; part-07 measures that.

**Closed gaps:**
- `Q01-applicable-only`: only `WARR-R2` reaches the model for a 2026 C-3 purchase. The 2025 policy and the S-2 scooter policy are filtered out before ranking.
- `Q01-historical`: a 2025 purchase gets `WARR-R1`, not the newer policy.
- `Q10-missing-details`: "How long is the battery warranty?" asks "Which product and purchase date is this about?". "What battery warranty applies to a bike bought on 15 March 2026?" asks only for the product, because policies for two products apply on that date.

**How earlier questions flow now:** the error-code, PIN, slide, and figure questions extract no date, so filtering only removes other products' documents and nothing asks for clarification. They retrieve as in part-05.

**Visible in the output:** the 2026 warranty answer also receives the battery safety checks table, because the system always fills five slots and nearest isn't relevant. Part 7 measures how much noise reaches the model; a relevance threshold is one remedy.

**Still can't:**
- List every safety check with a guarantee (`Q04`): retrieval returns the most relevant few. Part 9 builds a better source to query.
- Compute an inspection pass rate (`Q05`): Part 10.

**Stop here if** your users ask questions with one answer each, found in passages, tables, figures, or slides, including questions where the right document depends on a product, a date, or another stated condition. **This is the complete core.** Parts 7 and 8 measure it and watch it in production; Parts 9 to 13 are independent modules, so add only the ones your questions need.

## part-07: Measuring whether it works

**Article:** Measuring whether it works

**Lands in:** around the pipeline, plus two changes the measurements justified. The corpus grows from the 9-document sample used in Parts 1 to 6 to the full Atlas Bikes collection: 29 documents and 114 searchable pieces, with look-alike products (T-7 trekking, K-1 kids), an owner's manual, an FAQ, a brochure, dealer bulletins, and a price list only sales can read.

The check runner answers "does this case pass?". Measuring asks how well each stage works and where it is weak:

- `python -m atlas measure`: per question, whether a relevant block arrived (**hit**), how many of the relevant blocks arrived (**recall**), how much of what arrived is relevant (**precision**), the rank of the first relevant block, and the approximate prompt size in tokens. Then the same numbers by kind of question, and whether rewordings of key questions keep their result.
- `cases/held_out.json`: questions never used to tune the system. `measure --held-out` scores them separately.
- `python -m atlas judge-calibration`: a stand-in judge checks answers against their sources; its verdicts are compared with labels a person gave the same answers (`cases/judged_answers.json`).
- `measure --export-ragas FILE`: the same questions as a Ragas dataset (`user_input`, `retrieved_contexts`, `response`, `reference`).

**Growing the collection changed the test set too.** "Which battery check applies when the battery casing has been opened?" became ambiguous once the T-7 had its own check R07, so the case now names the C-3. "What does error E-417 mean?" names no bike, so the T-7's E-417 entry now counts as relevant too.

**What the held-out questions found.** Questions about the new documents missed things the development questions never showed:

- H12, "Which chargers are affected by the recall?", missed the recall bulletin. "Recall" is in the bulletin's title, and the searched forms from part-04 didn't include document titles. The searched forms now start with the document title. H12 was used to make that fix, so it moved into the development cases as `Q13-title-only`, and two fresh held-out questions replaced it.
- H10 ("How often should a T-7 be serviced?") and H17 ("What must be done before installing firmware 3.2?") still miss: "serviced" doesn't match "service", "installing" doesn't match "installation". That is a limitation of the stand-in embedding model; real embedding models handle word forms, but have blind spots of their own, which is what held-out questions are for.

By kind, held-out paraphrase questions have recall 0.75; everything else is 1.00.

**The threshold decision.** A minimum relevance score after reranking, tried with `measure --min-score` on the full collection:

| Minimum score | Development recall | Development precision | Held-out recall | Held-out precision |
| --- | --- | --- | --- | --- |
| 0 | 1.00 | 0.33 | 0.94 | 0.19 |
| 0.3 | 1.00 | 0.33 | 0.88 | 0.21 |
| 0.4 | 1.00 | 0.40 | 0.88 | 0.35 |
| 0.5 | 1.00 | 0.54 | 0.88 | 0.48 |
| 0.6 | 0.82 | 0.56 | 0.65 | 0.45 |

On the development questions, recall holds until 0.5 and drops at 0.6. The default is 0.4, chosen there with a margin. The held-out questions show its cost on new questions: H17's answer ranked second with a low score (0.25) and is cut by any threshold from 0.3 up. That is the expected price, about 1 question in 16, and it was not used to change the choice.

On the 9-document sample, the same threshold lifts development precision from 0.43 to 0.68 (measured with this checkpoint's code on the sample documents only). On the full collection it lifts it only from 0.33 to 0.40, because the remaining noise looks relevant: the FAQ's generic warranty answer, the frame warranty, the T-7's look-alike entries. A threshold can't separate those.

**What the judge showed:** it agreed with the person on 3 of 6 answers. It accepted "2 years" (the right answer is 3) and "eight checks" (there are eleven), because the words matched the sources; it rejected a correct answer phrased differently. A judge needs this kind of calibration before its scores mean anything.

**Still can't:** list every safety check with a guarantee (Part 9) or compute a pass rate (Part 10).

**Not a stopping point.** Like part-02, keep measuring whatever you build. Part 8 watches the same system once real users arrive.


## part-08: Watching it in production

**Article:** Watching it in production

**Lands in:** around the pipeline. Every stage now reports what it did; the answering logic doesn't change, apart from one fix the production log pointed to.

- Every request records a **span** per stage (field extraction, exact lookup, keyword search, vector search, fusion, reranking, the clarification check, assembly, the model call, citations), with timing and the details that explain the stage's result.
- `Assistant(log_path=...)` appends one line per request to `logs/requests.jsonl`: question, status, retrieved pieces and scores, supplied blocks, prompt size, answer, spans, and the settings and stand-ins in use. The user is stored as a pseudonym; feedback arrives later as separate events.
- `python -m atlas simulate` replays a week of staff and sales questions (`traffic/week-1.json`), written without looking at what the system would do with them.
- `python -m atlas stats`: outcomes, weak searches, withheld answers, feedback, time per stage, prompt size, products asked about, repeated questions, and a **review queue** of requests that deserve a person's look.
- `python -m atlas trace r007`: one request, stage by stage.
- `python -m atlas promote r007`: a draft test case in `cases/candidates.json` for a person to complete.

**What the week showed.** 24 requests, 22 answered and 2 clarifications, 4 thumbs up and 2 down. Four findings, of which the automatic signals caught only part:

- **r007, "What does E417 mean?" (thumbs down).** The trace shows `search.exact codes=[]`: typed without its dash, the code wasn't recognized, and the model received blink-pattern and display passages, all reranked at a confident 1.0. A high score didn't mean a right answer. **r020 asked the same question, left no feedback, and nothing flagged it.**
- **r010, a "C-5" bought in "May 2026" (thumbs down).** Atlas doesn't sell a C-5, so validation dropped it, and the stand-in extractor doesn't read month-and-year dates. The assistant asked for a product and a date the user had already given.
- **r006, a dealer price asked by service staff.** Access control worked: the price list never reached the model. But the assistant answered anyway, from four unrelated T-7 blocks scored 0.6. No signal flagged it; only reading a sample of requests would.
- **r008, the T-7 delivery checks**, flagged as a weak search (0.406), actually supplied the right table. Signals are hints for review, not verdicts.

**What changed.** The trace pointed at one stage, so the fix is small: codes are recognized with or without the dash and normalized to the catalog's form (`error_codes` in `pipeline.py`). r007 became `Q14-code-without-dash` in `cases.json`. r010 and r006 stay in `cases/candidates.json` with their expected behavior written down, accepted but not fixed yet.

The fix is a separate commit. `git checkout part-08~1` shows the system as it was during the week, and `python -m atlas simulate` then `python -m atlas trace r007` reproduces the failure.

**A note on timing.** The stand-ins answer in about 2 milliseconds, so the time per stage here only shows where time is measured. With a real embedding model, reranker, and language model, the model calls take most of the time, often seconds; that is what the same spans would reveal.

**Not a stopping point.** Keep logging, reviewing a sample, and promoting confirmed failures into test cases. Parts 9 to 13 are independent modules: add only the ones your questions need.

## part-09: When the question needs every match

**Article:** When the question needs every match

**Module.** Add it if your users ask for lists or counts: every error code, all the checks for a bike, how many of something there are.

**Lands in:** ingestion and the start of answering. Search itself doesn't change.

- **Records** (`atlas/records.py`): at ingestion, content with a fixed shape is also copied into records with fields. One per error code entry (`code`, `product`, `component`) and one per row of a battery safety check table (`check_id`, `product`). Each record keeps its document's access groups and points to its source section, so citations work as before.
- **A completeness report**: `python -m atlas records` compares the records with the documents, counted a second way, and flags codes mentioned anywhere without an entry. It writes `knowledge/records.json`, so a change in the documents shows up as a diff in review.
- **Curated records** (`knowledge/curated.json`): records a person writes, with their source and the reason they exist. The report found E-519, which firmware bulletin 2026-05 introduces before the error code documents list it; two curated records cover it until the next revision.
- **A router** (`Router` in `providers.py`, `ToyRouter` in `toy.py`): reads the question first and proposes a path, `search` or `list`, with a record type and an optional component. Code validates the proposal; anything it can't use falls back to search.
- **The list path**: every record of that type that the user may read and that fits the situation, not the best few. The prompt states how many there are.

**Answers:** "List every display error code" (10 entries, where search supplies 5), "How many error codes does the T-7 have?" (6, where search shows the model 5 because k is 5), "Which checks do I need before delivering a T-7?" (all 12), and Q04's eleven C-3 checks with completeness guaranteed.

**Measured:** 19 passing, 1 known gap, 0 regressions. In the replayed week, r008 (the Part 8 false alarm) now takes the list path, and there are no weak searches.

**Still can't:**
- Answer a list question about something that has no records. A question about "every bike affected by the recall" still goes to search.
- Guarantee more than the records do. The list is complete relative to the records, and the records are only as complete as their extraction and review.
- Compute anything from the records, such as a pass rate. That is Part 10.

**Stop here if** your list questions are about a few kinds of things with a fixed shape (codes, checks, products, contacts) and someone can own their records.

## part-10: When the answer must be calculated

**Article:** When the answer must be calculated

**Module.** Add it if your users ask for numbers that no document states: a rate, a total, an average over records.

**Lands in:** the records from part-09 and the router. Search doesn't change.

- **A new document**: `QA-INSPECT-2026`, final inspection results by batch. Search already finds its table for "What share of bikes passed final inspection without rework across batches A and B?"; the gap is the arithmetic. The table shows each batch's rate (80% and 90%), and averaging them gives 85%. The right answer adds the counts first: 89 of 110, 80.9%.
- **Inspection records** with numeric fields (`inspected`, `passed`), one per batch. The table's own pass rate column is not copied: rates are computed, never read. The completeness report flags batch F, whose passed count is blank.
- **Metrics defined in code** (`atlas/calculate.py`): `pass_rate` = sum of `passed` / sum of `inspected` over the inspection records in scope, with the document section that defines it. A blank count means no result (`incomplete_data`), an unknown batch is reported (`unknown_scope`), and a zero denominator gives no rate.
- **A third route**, `calculate`, with the metric and the batches named. "How is the pass rate defined?" stays on search.
- **The calculate path**: code computes the result; the model receives it as a sentence with the definition and the records used, and explains it. A **number check** then withholds any answer containing a number that appears nowhere in the result or the sources, such as an averaged 85%.

**Answers:** Q05 (80.9% for batches A and B), the T-7 rate (56 of 65, 86.2%), and refusals that say why: "What share of K-1 bikes passed?" can't be computed while batch F's count is blank.

**Measured:** 23 passing, 0 known gaps, 0 regressions. The remaining modules bring their own questions.

**Still can't:**
- Compute a metric nobody defined. Each metric is a deliberate addition: a formula, the records it reads, and tests.
- Answer questions whose evidence is spread across documents that refer to each other. That is Part 11.

**Stop here if** your numeric questions are about a known set of metrics over records you can extract and own.

## part-11: When the answer spans documents

**Article:** When the answer spans documents

**Module.** Add it if answers depend on documents that refer to each other through a shared name: a part number, a product, a customer, a contract.

**Lands in:** ingestion (a link graph) and a fourth route. Search doesn't change.

- **New evidence**: `BULLETIN-2026-09`, a battery recall that names only part BP-36-2. The specifications now give each bike's battery pack, the price list names them, and the S-2 warranty mentions `BP36-2`.
- **The gap**, on part-10 code: "Which products are affected by the battery recall?" supplies the recall and, next to it, the *charger* recall's list of affected models (C-3 and K-1). Nothing in the question leads to the C-3 specification or the S-2 warranty, so the right products never arrive, and a wrong list sits beside the evidence.
- **A link graph** (`atlas/graph.py`): every block is scanned for part numbers at ingestion; blocks that mention the same part are linked through it.
- **Entity resolution, simplest form**: part numbers are normalized before linking (`BP36-2` -> `BP-36-2`); without it, the S-2 is missed.
- **A fourth route**, `follow`, for "which bikes/models/products ..." questions: search as usual, take the best evidence, follow its part numbers one hop to every other block that mentions them, under the same access and situation filters. The prompt states the links followed.
- **Access applies to links**: the price list mentions BP-36-2 too, and only users in `sales` reach it.

**Answers:** the battery recall question with C-3 and S-2, each with the document that links it, and without the charger recall. "Which bikes use the BP-48-1 battery pack?" (held out).

**Measured:** 24 passing, 0 known gaps, 0 regressions.

**Still can't:**
- Choose where to start: links are followed from the best search result only. If that result is wrong, the chain is wrong.
- Follow more than one hop, or decide the next step from what it finds. That is Part 12.
- Link names that differ by more than spelling ("the removable 36 V battery" in the owner's manual). That needs an alias list or a model, with review.

**Stop here if** the links your questions need run through identifiers you can extract reliably (part numbers, product codes, contract IDs) and one hop is enough.

## part-12: When the next step depends on what you find

**Article:** When the next step depends on what you find

**Module.** Add it if some questions need a sequence of lookups whose next step depends on what the previous one found, and the fixed routes can't express that.

**Lands in:** a fifth route, `agent`, and recall records. The other routes don't change.

- **The gap**, on part-11 code: "A customer's C-3 has battery serial 2509-0412. Is it affected by a recall, and what should I do?" gets the battery recall, but not the C-3 specification that says the C-3 uses BP-36-2, nor the recall's "what to record" section, and the serial comparison is left to the model. The same question about a **T-7** gets the same recall, and the serial is inside its range: a model would very likely say "affected", but the T-7 uses BP-48-1.
- **Recall records**: a recall that states a serial range becomes a record per part (`part`, `first`, `last`), so a serial can be checked in code.
- **Tools** (`atlas/agent.py`): `search(query)`, `recall_status(part, serial)` (checked in code), `read(source_id)`, and `finish(evidence)` or `finish(clarify)`. Every tool applies the user's access.
- **The loop**: a planner (`Planner` in `providers.py`, `ToyPlanner` in `toy.py`) chooses each call after seeing the previous result. Code stops it after 6 steps or on a repeated call, and keeps only cited evidence that a tool actually showed. Each step is a span with the planner's stated reason, the call, and what it saw.
- **The answer** is built like the other routes: the cited evidence, the steps taken and the code-checked result as facts, then the number check and citations.

**Answers:** the affected C-3 (search, check, read: 5 model calls), a C-3 serial outside the range (search, check: 4), the T-7 (no recall for BP-48-1), and a question without a product (asks which product).

**Measured:** 26 passing, 0 known gaps, 0 regressions.

**Still can't:**
- Act: the tools only read. Creating the replacement order or updating the service record is a different problem, with different risks (a later series).
- Plan beyond its tools. A question that needs a tool nobody wrote gets a partial answer or none.
- Be cheaper than a route. Each step is a model call; send only the questions that need it.

**Stop here if** a handful of tools covers your multi-step questions, and the budget and guards are acceptable for the share of traffic that needs them.

## part-13: When the model should read whole documents

**Article:** When the model should read whole documents

**Module.** Add it if users ask about documents as a whole: what changed between revisions, how two documents differ, what a whole document says.

**Lands in:** a sixth route, `compare`.

- **The gap**, on part-12 code: "What changed between revision 1 and revision 2 of the T-7 battery warranty?" retrieves all four sections of both revisions, then asks "Which purchase date is this about?": Part 6's rule for date-scoped documents assumes a question about one situation. Even answered, the model would receive pieces in score order, not aligned, and nothing would say that the exclusions are identical.
- **Revision comparison** (`atlas/compare.py`): resolve both revisions by title (with the user's access), align sections by number and title, diff each pair in code word by word (`[-removed-] {+added+}`), and report unchanged sections too.
- **Whole documents, within a budget**: if both revisions fit `whole_budget` (2,000 tokens here), the model receives them whole; otherwise only the changed sections. Either way it receives the computed changes.
- **Prompt order**: documents and changes first, the question last, so a provider can reuse the long prefix across questions about the same documents (prompt caching; caching in general belongs to companion C, on speed and cost).

**Answers:** the T-7 comparison (term changed, exclusions unchanged), the C-3 comparison, and a refusal for a revision that doesn't exist.

**Measured:** 28 passing, 0 known gaps, 0 regressions. Held-out recall drops from 0.90 to 0.86: H23 ("How does revision 2 of the C-3 battery warranty differ from revision 1?") isn't recognized by the stand-in router's rule, goes to search, and is asked for a purchase date. Left as found: fixing the rule against a held-out question would make it a development question.

**Still can't:**
- Recognize comparison questions phrased differently from the stand-in rule (see H23). A model router would.
- Read a document larger than the budget whole. The diff keeps comparisons small; other whole-document questions need summarizing in parts or a recursive approach (see the article).
- Answer a question naming one revision without asking for a purchase date (Q25 is a routing test only).

**Stop here if** your whole-document questions are comparisons of versioned documents, or questions about documents that fit your model's context at an acceptable cost.

## part-14: Choosing your system

**Article:** Choosing your system

**Closing.** No new capability: two tools for deciding which parts you need.

- `python -m atlas needs FILE` routes a list of questions (a JSON file of requests or cases, or a text file with one question per line) and counts the paths they take, with the part of the series that builds each path.
- `Assistant(routes={...})` builds only the modules you choose; a question routed to a module you didn't build gets the core's search. `python -m atlas simulate --routes list` replays the week that way (`--routes none` is the core alone).

**What the tools show:** the dealer desk's week needs the core and Part 9 only (search 23, list 1). The test set needs every module, by design. The held-out set "needs" no comparison module, yet H23 is a comparison the stand-in router misses: route counts are a first look, to be checked by hand. With the core alone, r008 is a weak search again; with the list module, the week has none.

**Measured:** 28 passing, 0 known gaps, 0 regressions.

**Where to go from here:** companions A (access and trust), B (keeping the knowledge layer current), and C (speed and cost); a version of this repository with real models behind the same interfaces is planned.

