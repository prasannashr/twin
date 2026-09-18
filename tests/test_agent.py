"""Graph control flow: tool selection, budgets, validation, and request isolation."""
import json
import unittest
from unittest.mock import Mock, patch

from tests.helpers import call, completion, fake_research
from twin.agent import MAX_TOOL_ROUNDS, execute_readonly_tool, run_agent
from twin.retrieval import KnowledgeBase, Passage


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase([Passage('career', 'Intuit Java Kafka fraud risk 7,500 TPS'),
                                 Passage('leadership', 'Led six engineers at Javra')])

    def test_model_can_refine_search(self):
        model = Mock(side_effect=[completion(calls=[call(args={'query': 'xyznonexistent'})]),
                                 completion(calls=[call(args={'query': 'Java'})]),
                                 completion('Java experience. [S1]')])
        result = run_agent('Describe Java experience', [], model, self.kb)
        self.assertEqual(result['rounds'], 2)
        self.assertEqual(len(result['actions']), 2)
        self.assertIn('[S1]', result['answer'])

    def test_requirement_tool_keeps_gaps(self):
        sources = {}
        result = execute_readonly_tool('find_requirement_evidence', json.dumps({
            'requirements': ['Java', 'xyznonexistent']}), sources, self.kb)
        self.assertTrue(result['requirements'][0]['candidate_evidence'])
        self.assertEqual(result['requirements'][1]['candidate_evidence'], [])

    def test_forbidden_tools_and_invalid_arguments(self):
        for name, args in [('send_email', '{}'), ('search_profile', '{'),
                           ('search_profile', '{"query": 1}'),
                           ('search_profile', '{"query": "Java", "file": "/etc/passwd"}'),
                           ('find_requirement_evidence', '{"requirements": []}')]:
            self.assertIn('error', execute_readonly_tool(name, args, {}, self.kb))

    def test_private_tool_query_blocked(self):
        self.assertIn('error', execute_readonly_tool('search_profile', '{"query":"phone number"}', {}, self.kb))

    @patch('twin.legacy.notifications.requests.post')
    def test_legacy_contact_tool_cannot_execute(self, post):
        model = Mock(side_effect=[completion(calls=[call('record_user_details', {'email': 'visitor@example.com'})]),
                                 completion('Cannot send details.')])
        result = run_agent('Discuss career', [], model, self.kb)
        self.assertEqual(result['sources'], {})
        post.assert_not_called()

    def test_tool_loop_is_bounded(self):
        model = Mock(return_value=completion(calls=[call()]))
        result = run_agent('Discuss career', [], model, self.kb)
        self.assertEqual(model.call_count, MAX_TOOL_ROUNDS + 1)
        self.assertEqual(result['rounds'], MAX_TOOL_ROUNDS)
        self.assertEqual(model.call_args.kwargs['tool_choice'], 'none')

    def test_per_round_budget(self):
        model = Mock(side_effect=[completion(calls=[call(call_id=str(n)) for n in range(4)]), completion('Intuit [S1]')])
        result = run_agent('Discuss career', [], model, self.kb)
        self.assertEqual(len(result['actions']), 2)
        results = [m for m in model.call_args.args[0] if m['role'] == 'tool']
        self.assertEqual(len(results), 4)
        self.assertIn('budget', results[-1]['content'])

    def test_citations_must_exist(self):
        for answer in ('Invented reference [S99]', 'Uncited claim'):
            result = run_agent('Discuss career', [], Mock(side_effect=fake_research(answer)), self.kb)
            self.assertNotEqual(result['answer'], answer)

    def test_no_evidence_no_factual_answer(self):
        result = run_agent('Does he know Rust?', [], Mock(return_value=completion('Yes he does.')), self.kb)
        self.assertIn('not have enough', result['answer'])

    def test_greeting_needs_no_tools(self):
        result = run_agent('Hello', [], Mock(return_value=completion('Hello!')), self.kb)
        self.assertEqual(result['answer'], 'Hello!')
        self.assertEqual(result['rounds'], 0)

    def test_sources_do_not_leak_between_requests(self):
        first = run_agent('Discuss career', [], Mock(side_effect=fake_research()), self.kb)
        second = run_agent('Hello', [], Mock(return_value=completion('Hello!')), self.kb)
        self.assertTrue(first['sources'])
        self.assertEqual(second['sources'], {})


if __name__ == '__main__':
    unittest.main()
