# The Atlas Bikes corpus

Atlas Bikes is a fictional e-bike company. Every document, number, and rule here was written for teaching.

## What is in `parsed/`

Each JSON file is what a **layout-aware document parser** returns for one source file: a list of typed elements with page numbers. Real tools such as Docling, Unstructured, or Azure AI Document Intelligence produce richer output in a similar shape. We ship these outputs instead of the original PDFs and slides so the repository runs offline with no parser installed.

| File | Original | What makes it realistic |
| --- | --- | --- |
| `SPEC-C3-R5.json` | digital PDF | Repeated page headers and footers; the battery safety checks table split across pages 13 and 14, with no header on the second part; the table's scope printed in a paragraph above it. |
| `ASSEMBLY-C3-R3.json` | scanned PDF | OCR text with confidence scores; one paragraph read badly ("requlred", "dra1nage"). |
| `TEST-BAT-R1.json` | digital PDF | A figure whose values exist only in the image (`figures/TEST-BAT-R1-fig2.svg`). |
| `TRAIN-C3.json` | slide deck | Key information sits in speaker notes, not on the slides. |
| `ERR-C3-R2.json` | HTML page | Display error codes carried as heading attributes; two codes differ only by swapped digits (E-417, E-471). |
| `WARR-*.json`, `SERVICE-C3-R4.json` | Word documents | Clean headings and paragraphs: the easy case. Two battery warranty revisions apply to different purchase dates; a third covers the S-2 scooter. |

Document-level metadata (`product`, `access`, `applies_from`, `applies_through`) comes from the document management system, not from the text. Warranty policies are readable by all `employees`; the other documents by service `staff`.

## The full collection (from part-07)

Parts 1 to 6 work on the nine sample documents above. From part-07 the collection grows to 29 documents, the way a real company's would: look-alike products (the T-7 trekking bike and the K-1 kids' bike, each with its own warranty, specification, assembly guide, service manual, and the T-7 with its own error codes and training slides), the same topics written for different readers (a C-3 owner's manual, a customer FAQ, a brochure), dealer bulletins, a returns policy, a frame warranty for all bikes, a brake test report, and a parts price list that only `sales` can read. The nine sample documents are unchanged.

## Element types

```text
heading      text, level, optional attrs
paragraph    text
table        header (or null for a continuation), rows
figure       caption, image
slide        slide number, title, text, notes
page_header  text      (repeated page furniture)
page_footer  text
```

Scanned elements also carry `ocr_confidence`.

## Other files

- `figures/*.description.txt`: the description a vision model would produce for a figure. It stands in for that model's output so the examples run offline.
- `corrections.json`: corrections a person made after reviewing flagged OCR text (used from part-03).
