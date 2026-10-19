import unittest

from atlas.compare import word_diff
from atlas.pipeline import default_assistant

QUESTION = "What changed between revision 1 and revision 2 of the T-7 battery warranty?"


class Diff(unittest.TestCase):
    def test_word_diff_marks_removed_and_added_words(self):
        self.assertEqual(word_diff("a warranty of 2 years", "a warranty of 3 years"), "a warranty of [-2-] {+3+} years")

    def test_unchanged_sections_are_reported(self):
        diff = [s for s in default_assistant().ask(QUESTION).spans if s["stage"] == "diff"][0]
        self.assertEqual((diff["changed"], diff["unchanged"]), (["2"], ["3"]))


if __name__ == "__main__":
    unittest.main()
