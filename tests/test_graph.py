import unittest

from atlas.graph import part_numbers
from atlas.pipeline import STAFF, User, default_assistant


class PartNumbers(unittest.TestCase):
    def test_spelling_variants_resolve_to_one_part(self):
        self.assertEqual(part_numbers("part BP36-2"), {"BP-36-2"})
        self.assertEqual(part_numbers("packs BP-36-2 with serial numbers 2509-0001"), {"BP-36-2"})


class Links(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = default_assistant()
        cls.question = "Which products are affected by the battery recall?"

    def test_links_reach_documents_the_question_does_not(self):
        refs = [b.ref for b in self.assistant.ask(self.question).context]
        self.assertIn("SPEC-C3-R5#8", refs)
        self.assertIn("WARR-S2-R1#2", refs)

    def test_links_respect_access(self):
        self.assertNotIn("PRICE-PARTS-2026#1", [b.ref for b in self.assistant.ask(self.question, STAFF).context])
        manager = User("service manager", {"employees", "staff", "sales"})
        self.assertIn("PRICE-PARTS-2026#1", [b.ref for b in self.assistant.ask(self.question, manager).context])


if __name__ == "__main__":
    unittest.main()
