"""Local passage retrieval: no embedding service or vector database required."""
from twin.retrieval.knowledge_base import KnowledgeBase, Passage, tokenize
from twin.retrieval.loader import chunk_text, load_knowledge_base

__all__ = ['KnowledgeBase', 'Passage', 'tokenize', 'chunk_text', 'load_knowledge_base']
