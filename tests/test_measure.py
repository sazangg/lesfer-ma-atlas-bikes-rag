import unittest

from atlas.chunking import Chunk
from atlas.judge import ToyJudge
from atlas.measure import retrieval_row
from atlas.pipeline import Trace


def block(ref: str) -> Chunk:
    source_id, section = ref.split("#")
    return Chunk(ref, "text", source_id, section)


class RetrievalMetrics(unittest.TestCase):
    def test_hit_recall_precision_and_rank(self):
        trace = Trace("q", context=[block("A#1"), block("B#2"), block("C#3")], prompt="x" * 40)
        row = retrieval_row({"id": "c", "relevant": ["B#2", "D#4"]}, trace)
        self.assertEqual((row.hit, row.recall, row.first_relevant), (1.0, 0.5, 2))
        self.assertAlmostEqual(row.precision, 1 / 3)
        self.assertEqual(row.tokens, 10)

    def test_question_without_relevant_blocks_is_not_scored(self):
        row = retrieval_row({"id": "c", "relevant": []}, Trace("q", context=[block("A#1")]))
        self.assertIsNone(row.recall)


class Judge(unittest.TestCase):
    def test_flags_a_sentence_the_sources_dont_cover(self):
        judge = ToyJudge()
        sources = ["The battery is too hot to charge. Let it cool to room temperature."]
        self.assertEqual(judge.verdict("The battery is too hot to charge.", sources), "correct")
        self.assertEqual(judge.verdict("Replace the motor controller immediately.", sources), "wrong")


if __name__ == "__main__":
    unittest.main()
