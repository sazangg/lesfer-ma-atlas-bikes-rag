import unittest

from atlas.convert import convert
from atlas.corpus import ParsedDocument
from atlas.toy import ToyDescriber


def parsed(elements):
    return ParsedDocument("DOC", "doc.pdf", "pdf", {"access": "engineers"}, elements)


class Convert(unittest.TestCase):
    def test_removes_page_furniture(self):
        doc = convert(parsed([
            {"id": "e1", "type": "page_header", "page": 1, "text": "Company | Internal"},
            {"id": "e2", "type": "heading", "page": 1, "text": "1 Intro"},
            {"id": "e3", "type": "paragraph", "page": 1, "text": "Body."},
            {"id": "e4", "type": "page_footer", "page": 1, "text": "Page 1"},
        ]), ToyDescriber(), {})
        self.assertEqual(doc.report.removed_furniture, 2)
        self.assertEqual([e.render() for e in doc.sections[0].elements], ["Body."])

    def test_rejoins_table_continued_on_next_page_with_its_scope(self):
        doc = convert(parsed([
            {"id": "e1", "type": "heading", "page": 1, "text": "6 Rules"},
            {"id": "e2", "type": "paragraph", "page": 1, "text": "Scope: outdoor units."},
            {"id": "e3", "type": "table", "page": 1, "header": ["ID", "Rule"], "rows": [["R01", "a"]]},
            {"id": "e4", "type": "page_footer", "page": 1, "text": "Page 1"},
            {"id": "e5", "type": "table", "page": 2, "header": None, "rows": [["R02", "b"]]},
        ]), ToyDescriber(), {})
        table = doc.sections[0].elements[-1]
        self.assertEqual(table.data["rows"], [["R01", "a"], ["R02", "b"]])
        self.assertEqual(table.data["scope"], "Scope: outdoor units.")
        self.assertEqual(doc.report.merged_tables, 1)

    def test_flags_low_confidence_ocr_unless_corrected(self):
        elements = [{"id": "e1", "type": "paragraph", "page": 3, "text": "not requlred", "ocr_confidence": 0.6}]
        flagged = convert(parsed(elements), ToyDescriber(), {})
        self.assertEqual(len(flagged.report.needs_review), 1)
        fixed = convert(parsed(elements), ToyDescriber(), {"DOC": {"e1": "not required"}})
        self.assertEqual(fixed.report.needs_review, [])
        self.assertEqual(fixed.sections[0].elements[0].render(), "not required")

    def test_flags_gaps_in_numbered_rows(self):
        doc = convert(parsed([
            {"id": "e1", "type": "table", "page": 1, "header": ["ID", "Rule"], "rows": [["R01", "a"], ["R03", "c"]]},
        ]), ToyDescriber(), {})
        self.assertTrue(any("not consecutive" in note for note in doc.report.needs_review))


if __name__ == "__main__":
    unittest.main()
