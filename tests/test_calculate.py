import unittest

from atlas.calculate import calculate
from atlas.corpus import load_corpus
from atlas.pipeline import STAFF, Assistant, unsupported_numbers
from atlas.situation import Situation


class AveragingGenerator:
    """Stands in for a model that ignores the instructions and averages the batch rates."""

    def generate(self, prompt: str) -> str:
        return "The pass rate across batches A and B is 85% [S2][S3]."


class PassRate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = Assistant()
        cls.assistant.ingest(load_corpus())

    def rate(self, batches):
        return calculate("pass_rate", self.assistant.records, STAFF.groups, Situation(), batches)

    def test_counts_are_added_before_dividing(self):
        result = self.rate(["A", "B"])
        self.assertEqual((result["numerator"], result["denominator"], result["value"]), (89, 110, 80.9))

    def test_a_blank_count_gives_no_result(self):
        result = self.rate(["E", "F"])
        self.assertEqual((result["status"], result["incomplete"]), ("incomplete_data", ["F"]))
        self.assertNotIn("value", result)

    def test_an_unknown_batch_is_reported(self):
        self.assertEqual(self.rate(["G"])["status"], "unknown_scope")

    def test_numbers_not_in_the_prompt_are_caught(self):
        self.assertEqual(unsupported_numbers("about 85% [S1]", "89 of 110: 80.9%"), ["85"])

    def test_an_averaged_answer_is_withheld(self):
        assistant = Assistant(generator=AveragingGenerator())
        assistant.ingest(load_corpus())
        trace = assistant.ask("What share of bikes passed final inspection without rework across batches A and B?")
        self.assertEqual(trace.status, "number_mismatch")


if __name__ == "__main__":
    unittest.main()
