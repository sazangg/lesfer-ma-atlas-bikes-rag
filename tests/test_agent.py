import unittest

from atlas.agent import MAX_STEPS
from atlas.corpus import load_corpus
from atlas.pipeline import STAFF, Assistant, default_assistant

AFFECTED = "A customer's C-3 has battery serial 2509-0412. Is it affected by a recall, and what should I do?"


class Repeats:
    """A planner that keeps making the same call."""

    def next_step(self, question, steps, tools):
        return {"tool": "search", "args": {"query": "battery"}}


class Wanders:
    """A planner that never finishes."""

    def next_step(self, question, steps, tools):
        return {"tool": "search", "args": {"query": f"battery {len(steps)}"}}


class CitesUnseen:
    """A planner that finishes citing a source no tool showed it."""

    def next_step(self, question, steps, tools):
        return {"tool": "finish", "args": {"evidence": ["PRICE-PARTS-2026#1"]}}


class TriesPriceList:
    def next_step(self, question, steps, tools):
        if not steps:
            return {"tool": "read", "args": {"source_id": "PRICE-PARTS-2026"}}
        return {"tool": "finish", "args": {"evidence": ["PRICE-PARTS-2026#1"]}}


def with_planner(planner) -> Assistant:
    assistant = Assistant(planner=planner)
    assistant.ingest(load_corpus())
    return assistant


class Paths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = default_assistant()

    def tools_used(self, question):
        return [s["tool"] for s in self.assistant.ask(question).spans if s["stage"].startswith("step")]

    def test_the_path_depends_on_what_each_step_finds(self):
        self.assertEqual(self.tools_used(AFFECTED), ["search", "recall_status", "read", "finish"])
        self.assertEqual(self.tools_used(AFFECTED.replace("0412", "0950")), ["search", "recall_status", "finish"])

    def test_a_missing_product_is_asked_for(self):
        trace = self.assistant.ask("Battery serial 2509-0412: is it affected by a recall?")
        self.assertEqual((trace.status, trace.answer), ("clarify", "Which product is this about?"))


class Guards(unittest.TestCase):
    def test_a_repeated_call_stops_the_loop(self):
        self.assertEqual(with_planner(Repeats()).ask(AFFECTED).status, "gave_up")

    def test_the_step_budget_stops_the_loop(self):
        trace = with_planner(Wanders()).ask(AFFECTED)
        self.assertEqual(trace.status, "gave_up")
        self.assertEqual(sum(s["stage"].startswith("step") for s in trace.spans), MAX_STEPS)

    def test_only_what_a_tool_showed_can_be_cited(self):
        trace = with_planner(CitesUnseen()).ask(AFFECTED)
        self.assertEqual(trace.context, [])

    def test_tools_apply_access(self):
        trace = with_planner(TriesPriceList()).ask(AFFECTED, STAFF)
        self.assertEqual(trace.context, [])


if __name__ == "__main__":
    unittest.main()
