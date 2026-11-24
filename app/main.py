from fastapi import FastAPI, Query

from app.ingest import ingest
from app.tfidf import TfidfRetriever

app = FastAPI(
    title="Vector RAG Desk",
    summary="Embeddings RAG with citations, abstention, and a TF-IDF baseline on the same gold set.",
)

_retriever: TfidfRetriever | None = None


def get_retriever() -> TfidfRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TfidfRetriever(ingest())
    return _retriever


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest_corpus() -> dict[str, int]:
    global _retriever
    chunks = ingest()
    _retriever = TfidfRetriever(chunks)
    return {"chunks": len(chunks)}


@app.get("/retrieve")
def retrieve(
    q: str = Query(..., min_length=1),
    k: int = Query(4, ge=1, le=10),
) -> dict[str, object]:
    matches = [
        {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "text": chunk.text,
            "score": score,
        }
        for chunk, score in get_retriever().search(q, limit=k)
    ]
    return {"retriever": "tfidf", "query": q, "matches": matches}
