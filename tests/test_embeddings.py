from pathlib import Path

from fastapi.testclient import TestClient

from app.embedder import FakeEmbedder
from app.ingest import ingest
from app.main import app
from app.settings import EMBEDDINGS_ABSTAIN_THRESHOLD, INDEX_PATH
from app.vector_index import VectorIndex

client = TestClient(app)


def test_fake_embedder_is_deterministic_and_normalized() -> None:
    embedder = FakeEmbedder()
    first = embedder.embed(["dispatch lead reviews the queue at 17:00"])[0]
    second = embedder.embed(["dispatch lead reviews the queue at 17:00"])[0]
    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_vector_index_ranks_relevant_chunk_above_unrelated() -> None:
    chunks = ingest()
    embedder = FakeEmbedder()
    index = VectorIndex.build(chunks, embedder.embed([chunk.text for chunk in chunks]))
    hits = index.search(
        embedder.embed(["What time does the dispatch lead review the next-day order queue?"])[0],
        limit=20,
    )
    assert hits
    assert hits[0][0].doc_id == "dispatch-planning"
    weather = next(score for chunk, score in hits if chunk.doc_id == "weather-and-access-safety")
    assert hits[0][1] > weather


def test_ingest_persists_sqlite_vec_index() -> None:
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    response = client.post("/ingest")
    assert response.status_code == 200
    payload = response.json()
    assert payload["indexed"] >= 8
    assert payload["embedder"] == "fake"
    assert Path(INDEX_PATH).exists()


def test_retrieve_embeddings_returns_threshold() -> None:
    client.post("/ingest")
    response = client.get(
        "/retrieve",
        params={"q": "When may a driver pause a stop for severe weather?", "retriever": "embeddings"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["retriever"] == "embeddings"
    assert payload["abstain_threshold"] == EMBEDDINGS_ABSTAIN_THRESHOLD
    assert payload["matches"][0]["doc_id"] == "weather-and-access-safety"
    assert "would_abstain" in payload
