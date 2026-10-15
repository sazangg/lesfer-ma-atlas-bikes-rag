import unittest

from atlas.pipeline import error_codes


class ErrorCodes(unittest.TestCase):
    def test_codes_with_and_without_a_dash_match_the_catalog_form(self):
        self.assertEqual(error_codes("What does E-417 mean?"), {"E-417"})
        self.assertEqual(error_codes("What does E417 mean?"), {"E-417"})

    def test_ordinary_words_are_not_codes(self):
        self.assertEqual(error_codes("How do I unlock the display?"), set())


if __name__ == "__main__":
    unittest.main()
