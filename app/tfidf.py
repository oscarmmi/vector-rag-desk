from __future__ import annotations

import math
import re
from collections import Counter

from app.ingest import Chunk

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def term_counts(text: str) -> dict[str, int]:
    tokens = [token.lower() for token in TOKEN.findall(text) if token.lower() not in STOP_WORDS]
    return dict(Counter(tokens))


def _weighted(counts: dict[str, int], document_frequency: dict[str, int], corpus_size: int) -> dict[str, float]:
    vector: dict[str, float] = {}
    for term, count in counts.items():
        idf = math.log((corpus_size + 1) / (document_frequency.get(term, 0) + 1)) + 1
        vector[term] = count * idf
    return vector


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._term_lists = [term_counts(chunk.text) for chunk in chunks]
        self._document_frequency: dict[str, int] = {}
        for terms in self._term_lists:
            for term in set(terms):
                self._document_frequency[term] = self._document_frequency.get(term, 0) + 1
        self._corpus_size = len(chunks)

    def search(self, query: str, limit: int = 4) -> list[tuple[Chunk, float]]:
        query_terms = term_counts(query)
        if not self.chunks or not query_terms:
            return []
        query_vector = _weighted(query_terms, self._document_frequency, self._corpus_size)
        scored: list[tuple[int, float]] = []
        for index, terms in enumerate(self._term_lists):
            document_vector = _weighted(terms, self._document_frequency, self._corpus_size)
            score = _cosine(query_vector, document_vector)
            scored.append((index, round(score, 4)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(self.chunks[index], score) for index, score in scored[: max(0, limit)]]
