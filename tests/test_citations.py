import unittest

from atlas.pipeline import resolve_citations


class ResolveCitations(unittest.TestCase):
    def test_known_label_becomes_link(self):
        answer, unknown = resolve_citations("24 months. [S1]", {"S1": "WARR-R2#2"})
        self.assertEqual(answer, "24 months. [WARR-R2, section 2](/documents/WARR-R2#section-2)")
        self.assertEqual(unknown, [])

    def test_unknown_label_is_reported_not_linked(self):
        answer, unknown = resolve_citations("24 months. [S3]", {"S1": "WARR-R2#2"})
        self.assertEqual(unknown, ["S3"])
        self.assertIn("[S3]", answer)


if __name__ == "__main__":
    unittest.main()
