"""Index construction and BM25 ranking."""
import unittest

from twin.retrieval import KnowledgeBase, Passage, load_knowledge_base


class RetrievalTests(unittest.TestCase):
    def test_retrieval_ranks_matching_passage(self):
        kb = KnowledgeBase([Passage('a', 'Java Kafka distributed systems'), Passage('b', 'Hiking trekking nature')])
        self.assertEqual(kb.search('Kafka')[0].source, 'a')
        self.assertEqual(kb.search('quantum entanglement'), [])

    def test_profile_retrieval(self):
        hits = load_knowledge_base().search('Intuit fraud risk 7,500 TPS')
        self.assertTrue(any('7,500' in p.text for p in hits))


if __name__ == '__main__':
    unittest.main()
