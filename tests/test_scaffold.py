import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.corpus import document_ids
from app.main import app

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "gold.json"

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["embeddings_abstain_threshold"] == 0.35
    assert payload["tfidf_abstain_threshold"] == 0.08


def test_gold_set_has_at_least_30_questions() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    questions = payload["questions"]
    assert len(questions) >= 30


def test_gold_expected_docs_exist_in_corpus() -> None:
    known = document_ids()
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    for item in payload["questions"]:
        expected = item["expected_doc"]
        if item["should_abstain"]:
            assert expected is None
            continue
        assert expected in known, f"{item['id']} points at missing doc {expected}"
