from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from app.ingest import Chunk
from app.settings import EMBEDDING_DIM, INDEX_PATH


class VectorIndex:
    def __init__(self, path: Path, chunks: list[Chunk]) -> None:
        self.path = path
        self.chunks = chunks
        self._by_rowid = {index + 1: chunk for index, chunk in enumerate(chunks)}

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: list[list[float]], path: Path = INDEX_PATH) -> VectorIndex:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        connection = _connect(path)
        try:
            connection.execute(
                f"""
                CREATE VIRTUAL TABLE vec_chunks USING vec0(
                    embedding float[{EMBEDDING_DIM}] distance_metric=cosine
                )
                """
            )
            for index, vector in enumerate(embeddings):
                connection.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    [index + 1, serialize_float32(vector)],
                )
            connection.commit()
        finally:
            connection.close()
        return cls(path, chunks)

    def search(self, query_vector: list[float], limit: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        connection = _connect(self.path)
        try:
            rows = connection.execute(
                """
                SELECT rowid, distance
                FROM vec_chunks
                WHERE embedding MATCH ?
                  AND k = ?
                ORDER BY distance
                """,
                [serialize_float32(query_vector), max(1, limit)],
            ).fetchall()
        finally:
            connection.close()
        matches: list[tuple[Chunk, float]] = []
        for rowid, distance in rows:
            chunk = self._by_rowid.get(int(rowid))
            if chunk is None:
                continue
            similarity = round(1.0 - float(distance), 4)
            matches.append((chunk, similarity))
        return matches


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection
