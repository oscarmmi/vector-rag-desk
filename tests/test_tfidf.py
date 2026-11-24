from fastapi.testclient import TestClient

from app.chunker import split_text
from app.ingest import ingest
from app.main import app
from app.tfidf import TfidfRetriever

client = TestClient(app)


def test_chunker_splits_long_text_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(80))
    chunks = split_text(text, chunk_size=120, overlap=20)
    assert len(chunks) >= 2
    first_tail = chunks[0].split()[-3:]
    second_tokens = chunks[1].split()
    assert any(token in second_tokens for token in first_tail)


def test_ingest_writes_chunks_for_every_handbook_page() -> None:
    chunks = ingest()
    doc_ids = {chunk.doc_id for chunk in chunks}
    assert len(chunks) >= 8
    assert "dispatch-planning" in doc_ids
    assert "weather-and-access-safety" in doc_ids


def test_relevant_chunk_ranks_above_unrelated() -> None:
    retriever = TfidfRetriever(ingest())
    hits = retriever.search(
        "What time does the dispatch lead review the next-day order queue?",
        limit=20,
    )
    assert hits
    assert hits[0][0].doc_id == "dispatch-planning"
    weather = next(score for chunk, score in hits if chunk.doc_id == "weather-and-access-safety")
    assert hits[0][1] > weather


def test_retrieve_endpoint_returns_tfidf_matches() -> None:
    ingest_response = client.post("/ingest")
    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunks"] >= 8

    response = client.get("/retrieve", params={"q": "When may a driver pause a stop for severe weather?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["retriever"] == "tfidf"
    assert payload["matches"][0]["doc_id"] == "weather-and-access-safety"
    assert payload["matches"][0]["score"] > 0
