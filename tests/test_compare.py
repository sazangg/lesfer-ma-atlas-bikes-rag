import unittest

from atlas.cache import AnswerCache
from atlas.compare import word_diff
from atlas.pipeline import SALES, STAFF, default_assistant

QUESTION = "What changed between revision 1 and revision 2 of the T-7 battery warranty?"


class Diff(unittest.TestCase):
    def test_word_diff_marks_removed_and_added_words(self):
        self.assertEqual(word_diff("a warranty of 2 years", "a warranty of 3 years"), "a warranty of [-2-] {+3+} years")

    def test_unchanged_sections_are_reported(self):
        diff = [s for s in default_assistant().ask(QUESTION).spans if s["stage"] == "diff"][0]
        self.assertEqual((diff["changed"], diff["unchanged"]), (["2"], ["3"]))


class Cache(unittest.TestCase):
    def setUp(self):
        self.assistant = default_assistant()
        self.assistant.cache = AnswerCache()

    def test_a_repeated_question_is_answered_from_the_cache(self):
        self.assistant.ask("What does E-417 mean?")
        self.assertTrue(self.assistant.ask("what does e-417 mean").cached)

    def test_answers_are_not_shared_across_access_groups(self):
        self.assistant.ask("What is the dealer price of a T-7 battery?", SALES)
        self.assertFalse(self.assistant.ask("What is the dealer price of a T-7 battery?", STAFF).cached)

    def test_a_thumbs_down_evicts_the_answer(self):
        self.assistant.ask("What does E-417 mean?")
        self.assistant.forget("What does E-417 mean?", STAFF)
        self.assertFalse(self.assistant.ask("What does E-417 mean?").cached)


if __name__ == "__main__":
    unittest.main()
