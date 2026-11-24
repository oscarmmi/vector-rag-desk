from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

from app.settings import EMBEDDING_DIM, EMBEDDING_MODEL
from app.tfidf import STOP_WORDS, TOKEN


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors. Used in pytest (no torch)."""

    name = "fake"
    dim = EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]


class SentenceTransformerEmbedder:
    name = "minilm"
    dim = EMBEDDING_DIM

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.astype(float).tolist() for vector in vectors]


def get_embedder() -> Embedder:
    choice = os.environ.get("VECTOR_RAG_EMBEDDER", "").strip().lower()
    if choice == "fake":
        return FakeEmbedder()
    if choice == "minilm":
        return SentenceTransformerEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        return FakeEmbedder()


def _hash_vector(text: str) -> list[float]:
    values = [0.0] * EMBEDDING_DIM
    for token in TOKEN.findall(text.lower()):
        if token in STOP_WORDS:
            continue
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
        values[index] += 1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
