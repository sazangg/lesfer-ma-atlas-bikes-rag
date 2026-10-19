"""Part 13: when the model should read whole documents.

"What changed between revision 1 and revision 2 of the T-7 battery warranty?"
is a question about two documents, not about a situation. Search returns
pieces; a comparison needs every section of both, side by side, including the
sections that didn't change. Saying "nothing else changed" requires having
read everything else.

So the compare path:

    resolve both revisions by title -> align sections by number and title
    -> diff each section in code, word by word -> supply the documents
    (whole if they fit the budget, changed sections only if not) + the diff
    -> the model explains the changes

The diff is computed, not generated, for the same reason the pass rate is:
a model comparing two long texts by eye misses small changes, and the
smallest change ("or 1,000 full charge cycles") can matter most.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from atlas.chunking import Chunk
from atlas.index import readable


# Kinds of document a question can ask to compare, as they appear in document titles.
DOCUMENT_KINDS = ["battery warranty", "specification", "service manual", "assembly guide", "display error codes"]


@dataclass
class SectionChange:
    number: str
    title: str
    status: str  # changed, added, removed, unchanged
    diff: str = ""  # word diff: [-removed-] {+added+}


def find_revision(documents, product: str | None, kind: str, revision: str, groups: set[str]):
    """The document whose title names this product, kind, and revision, if the user may read it."""
    for doc in documents:
        title = doc.meta.get("title", "")
        if (kind in title.lower() and (not product or title.startswith(product))
                and title.endswith(f"revision {revision}") and readable(Chunk("", "", "", "", doc.meta), groups)):
            return doc
    return None


def section_text(section) -> str:
    return " ".join(e.render() for e in section.elements)


def word_diff(old: str, new: str) -> str:
    a, b = old.split(), new.split()
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            parts.append(" ".join(a[i1:i2]))
        if op in ("delete", "replace"):
            parts.append("[-" + " ".join(a[i1:i2]) + "-]")
        if op in ("insert", "replace"):
            parts.append("{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(parts)


def compare(old_doc, new_doc) -> list[SectionChange]:
    """Align sections by number and title, then diff each pair."""
    key = lambda s: (s.number, s.title)
    old = {key(s): s for s in old_doc.sections if s.number}
    new = {key(s): s for s in new_doc.sections if s.number}
    changes = []
    for k in sorted(old.keys() | new.keys(), key=lambda k: [int(x) for x in re.findall(r"\d+", k[0])]):
        number, title = k
        if k not in new:
            changes.append(SectionChange(number, title, "removed", section_text(old[k])))
        elif k not in old:
            changes.append(SectionChange(number, title, "added", section_text(new[k])))
        elif section_text(old[k]) == section_text(new[k]):
            changes.append(SectionChange(number, title, "unchanged"))
        else:
            changes.append(SectionChange(number, title, "changed", word_diff(section_text(old[k]), section_text(new[k]))))
    return changes


def render(changes: list[SectionChange]) -> list[str]:
    return [f"Section {c.number} ({c.title}): {c.status}" + (f": {c.diff}" if c.diff else "") for c in changes]
