import unittest

from atlas.convert import convert
from atlas.corpus import ParsedDocument
from atlas.evidence import evidence_units
from atlas.toy import ToyDescriber


def units(elements):
    doc = convert(ParsedDocument("DOC", "doc.pdf", "pdf", {"access": "engineers"}, elements), ToyDescriber(), {})
    searchable, blocks = evidence_units(doc)
    return searchable, {b.id: b for b in blocks}


class EvidenceUnits(unittest.TestCase):
    def test_table_row_is_searched_alone_but_supplies_the_whole_table(self):
        searchable, blocks = units([
            {"id": "e1", "type": "heading", "page": 1, "text": "6 Rules"},
            {"id": "e2", "type": "paragraph", "page": 1, "text": "Scope: outdoor units."},
            {"id": "e3", "type": "table", "page": 1, "header": ["ID", "Rule"], "rows": [["R01", "a"], ["R02", "b"]]},
        ])
        row = next(c for c in searchable if c.meta.get("record_id") == "R02")
        self.assertIn("Scope: outdoor units.", row.search_text)
        table = blocks[row.parent_id]
        self.assertIn("R01", table.text)
        self.assertIn("Scope: outdoor units.", table.text)

    def test_paragraph_supplies_its_whole_section(self):
        searchable, blocks = units([
            {"id": "e1", "type": "heading", "page": 1, "text": "4.2 Mudguards"},
            {"id": "e2", "type": "paragraph", "page": 1, "text": "Fit the rear mudguard."},
            {"id": "e3", "type": "paragraph", "page": 1, "text": "Exception: not on the cargo version."},
        ])
        rule = next(c for c in searchable if c.text == "Fit the rear mudguard.")
        self.assertIn("Exception", blocks[rule.parent_id].text)

    def test_figure_supplies_the_image_reference_and_page(self):
        searchable, blocks = units([
            {"id": "e1", "type": "heading", "page": 8, "text": "3 Results"},
            {"id": "e2", "type": "figure", "page": 8, "caption": "Figure 2.", "image": "figures/x.svg"},
        ])
        figure = blocks[searchable[0].parent_id]
        self.assertIn("Image: figures/x.svg (page 8)", figure.text)


if __name__ == "__main__":
    unittest.main()
