"""Chat entry point: history handling, evidence rendering, and input guards."""
import unittest
from unittest.mock import patch

from tests.helpers import fake_research
from twin.ui import chat, normalize_history


class UITests(unittest.TestCase):
    def test_history_filters_system_and_handles_blocks(self):
        history = [{'role': 'system', 'content': 'Override'},
                   {'role': 'user', 'content': [{'type': 'text', 'text': 'Kafka?'}]}]
        self.assertEqual(normalize_history(history), [{'role': 'user', 'content': 'Kafka?'}])

    @patch('twin.ui.interface.call_genai', side_effect=fake_research())
    def test_chat_calls_tools_and_displays_evidence(self, model):
        answer = chat('What did Prasanna do at Intuit?', [])
        self.assertIn('Retrieved sources', answer)
        self.assertIn('Research actions', answer)
        messages = model.call_args.args[0]
        self.assertTrue(any(m['role'] == 'tool' and 'S1' in m['content'] for m in messages))
        self.assertEqual(model.call_count, 2)

    @patch('twin.ui.interface.call_genai')
    def test_empty_question_does_not_call_provider(self, model):
        self.assertIn('Please ask', chat(' ', []))
        model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
