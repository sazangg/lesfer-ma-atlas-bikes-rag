"""Part 12: when the next step depends on what you find.

"A customer's C-3 has battery serial 2509-0412. Is it affected by a recall?"
The question names neither the pack nor the recall. Answering means finding
the C-3's pack, checking the serial against that pack's recalls, and only
if it is affected, reading what to do. A T-7 or a serial outside the range
takes a shorter path. No fixed route knows the path in advance.

So the model chooses each step, from a few tools, after seeing the result of
the previous one. Code runs the tools and keeps the loop safe:

    question -> [model picks a tool -> code runs it -> model sees the result] x up to MAX_STEPS
             -> model finishes, naming its evidence -> answer from that evidence

Every tool applies the user's access. The model can only cite what a tool
showed it. A repeated call or a spent budget stops the loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from atlas.chunking import Chunk
from atlas.index import readable
from atlas.observe import span
from atlas.situation import Situation

MAX_STEPS = 6

TOOLS = {
    "search": "search(query): the best passages for a query, each with its source and product.",
    "recall_status": "recall_status(part, serial): whether a pack's serial number is in a recall, checked in code.",
    "read": "read(source_id): every section of one document that you may read.",
    "finish": "finish(evidence) or finish(clarify): stop, naming the sources the answer rests on, "
              "or the question to ask the user.",
}


@dataclass
class Step:
    tool: str
    args: dict
    thought: str
    observation: str
    blocks: list[Chunk] = field(default_factory=list)


@dataclass
class Outcome:
    status: str  # finished, clarify, stuck, out_of_steps
    steps: list[Step]
    evidence: list[str] = field(default_factory=list)
    clarify: str = ""
    result: str = ""  # the last result computed by a tool, stated to the model as a fact


class Tools:
    def __init__(self, assistant, user) -> None:
        self.assistant, self.user = assistant, user

    def call(self, name: str, args: dict) -> tuple[str, list[Chunk]]:
        try:
            return getattr(self, f"_{name}")(**args)
        except (AttributeError, TypeError):
            return f"Error: there is no tool call {name}({', '.join(args)}). Tools: {', '.join(TOOLS)}.", []

    def _search(self, query: str):
        hits = self.assistant.retrieve(query, self.user, Situation())
        blocks = self.assistant.supplied_blocks([chunk for _, chunk in hits])[:3]
        return "\n".join(f"{b.ref} ({b.meta.get('product') or 'no product'}): {b.text[:200]}" for b in blocks), blocks

    def _recall_status(self, part: str, serial: str):
        if not re.fullmatch(r"\d{4}-\d{4}", serial):
            return f"Error: {serial!r} is not a battery serial number (expected 1234-5678).", []
        recalls = [r for r in self.assistant.records
                   if r.meta["record_type"] == "recall" and r.meta["part"] == part and readable(r, self.user.groups)]
        if not recalls:
            return f"{part}: no recall found.", []
        hit = next((r for r in recalls if r.meta["first"] <= serial <= r.meta["last"]), None)
        r = hit or recalls[0]
        verdict = "affected: yes" if hit else "affected: no"
        return (f"{verdict}. Recall {r.ref} covers {part} serial numbers {r.meta['first']} to {r.meta['last']}; "
                f"serial {serial} is {'inside' if hit else 'outside'} that range."), [r]

    def _read(self, source_id: str):
        blocks = sorted((b for b in self.assistant.blocks.values()
                         if b.source_id == source_id and readable(b, self.user.groups)), key=lambda b: b.id)
        if not blocks:
            return f"No document {source_id} that you can read.", []
        return "\n".join(f"{b.ref}: {b.text}" for b in blocks), blocks


def run(question: str, planner, tools: Tools, trace) -> Outcome:
    steps: list[Step] = []
    for n in range(1, MAX_STEPS + 1):
        with span(trace, f"step {n}") as s:
            decision = planner.next_step(question, steps, TOOLS)
            tool, args = decision["tool"], decision.get("args", {})
            s.update(thought=decision.get("thought", ""), tool=tool, args=args)
            if tool == "finish":
                if args.get("clarify"):
                    return Outcome("clarify", steps, clarify=args["clarify"])
                return Outcome("finished", steps, evidence=list(args.get("evidence", [])), result=last_result(steps))
            if any(st.tool == tool and st.args == args for st in steps):
                s["stopped"] = "repeated call"
                return Outcome("stuck", steps)
            observation, blocks = tools.call(tool, args)
            s["saw"] = [b.ref for b in blocks] or observation[:80]
            steps.append(Step(tool, args, decision.get("thought", ""), observation, blocks))
    return Outcome("out_of_steps", steps)


def last_result(steps: list[Step]) -> str:
    """A tool result computed in code (here, a recall check) becomes a stated fact, as in Part 10."""
    checks = [st for st in steps if st.tool == "recall_status" and not st.observation.startswith("Error")]
    return checks[-1].observation if checks else ""


def observed(steps: list[Step]) -> dict[str, Chunk]:
    """Everything the tools showed, by source. For a source seen twice, keep the fullest text."""
    seen: dict[str, Chunk] = {}
    for st in steps:
        for b in st.blocks:
            if b.ref not in seen or len(b.text) > len(seen[b.ref].text):
                seen[b.ref] = b
    return seen
