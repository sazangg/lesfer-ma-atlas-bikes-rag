import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from atlas.corpus import load_corpus
from atlas.needs import read_questions
from atlas.pipeline import Assistant


class BuildOnlyWhatYouNeed(unittest.TestCase):
    def test_a_module_you_did_not_build_falls_back_to_search(self):
        core = Assistant(routes=set())
        core.ingest(load_corpus())
        trace = core.ask("List every display error code.")
        self.assertEqual(trace.route, "search")
        self.assertEqual([s.get("not_built") for s in trace.spans if s["stage"] == "route"], ["list"])

    def test_questions_can_come_from_a_text_file(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "questions.txt"
            path.write_text("What does E-417 mean?\n\nList every display error code.\n", encoding="utf-8")
            self.assertEqual(len(read_questions(path)), 2)


if __name__ == "__main__":
    unittest.main()
