"""BM25 passage index held in the server process's memory."""
from collections import Counter
from dataclasses import dataclass
from math import log
import re

from twin.privacy import redact_private_data

STOP_WORDS = set('a an the is are was were be been to of in on at for and or with me my you your i he his him it its about what which how tell does do did can has have who'.split())

DEFAULT_SEARCH_LIMIT = 4
# Okapi BM25 saturation and length-normalisation parameters.
K1 = 1.5
B = 0.75


def tokenize(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP_WORDS]


@dataclass(frozen=True)
class Passage:
    source: str
    text: str


class KnowledgeBase:
    def __init__(self, passages):
        passages = [Passage(p.source, redact_private_data(p.text)) for p in passages]
        self.passages = passages
        self.counts = [Counter(tokenize(p.text)) for p in passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average_length = sum(self.lengths) / max(1, len(passages)) or 1
        self.document_frequency = Counter(word for c in self.counts for word in c)

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        """BM25 lexical ranking; return only passages with matching terms."""
        return [passage for _, passage in self.ranked(query)[:limit]]

    def ranked(self, query):
        """Score every matching passage, highest first. Used by search and the eval suite."""
        terms = set(tokenize(query))
        scored = []
        for passage, counts, length in zip(self.passages, self.counts, self.lengths):
            score = 0
            for term in terms:
                frequency = counts[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                inverse_frequency = log(1 + (len(self.passages) - df + 0.5) / (df + 0.5))
                score += inverse_frequency * frequency * (K1 + 1) / (
                    frequency + K1 * (1 - B + B * length / self.average_length)
                )
            if score > 0:
                scored.append((score, passage))
        return sorted(scored, key=lambda item: item[0], reverse=True)
