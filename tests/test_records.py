import unittest

from atlas.pipeline import STAFF, User, default_assistant, validate_route
from atlas.records import component_of
from atlas.toy import ToyRouter


class Routing(unittest.TestCase):
    def test_list_questions_take_the_list_path(self):
        route = ToyRouter().route("List every display error code.")
        self.assertEqual((route["path"], route["record_type"]), ("list", "error_code"))

    def test_a_single_answer_which_question_stays_on_search(self):
        self.assertEqual(ToyRouter().route("Which error code means the battery is too hot?")["path"], "search")

    def test_an_unknown_record_type_falls_back_to_search(self):
        self.assertEqual(validate_route({"path": "list", "record_type": "warranty"})["path"], "search")


class Records(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = default_assistant()

    def test_component_is_the_first_component_word_in_the_meaning(self):
        self.assertEqual(component_of("the battery is too hot to charge."), "battery")
        self.assertEqual(component_of("the fast charger reported a voltage it doesn't support."), "charger")

    def test_list_path_returns_every_match_beyond_k(self):
        trace = self.assistant.ask("List every display error code.")
        self.assertGreater(len(trace.context), self.assistant.k)
        self.assertEqual(trace.complete_by, "enumeration")

    def test_records_keep_their_document_access(self):
        outsider = User("visitor", {"public"})
        self.assertEqual(self.assistant.ask("List every display error code.", outsider).context, [])
        self.assertTrue(self.assistant.ask("List every display error code.", STAFF).context)


if __name__ == "__main__":
    unittest.main()
