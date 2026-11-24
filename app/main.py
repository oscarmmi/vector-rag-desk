from fastapi import FastAPI, Query

from app.embedder import Embedder, get_embedder
from app.ingest import Chunk, ingest
from app.settings import EMBEDDINGS_ABSTAIN_THRESHOLD, TFIDF_ABSTAIN_THRESHOLD
from app.tfidf import TfidfRetriever
from app.vector_index import VectorIndex

app = FastAPI(
    title="Vector RAG Desk",
    summary="Embeddings RAG with citations, abstention, and a TF-IDF baseline on the same gold set.",
)

_chunks: list[Chunk] | None = None
_tfidf: TfidfRetriever | None = None
_index: VectorIndex | None = None
_embedder: Embedder | None = None


def _rebuild() -> tuple[list[Chunk], TfidfRetriever, VectorIndex, Embedder]:
    global _chunks, _tfidf, _index, _embedder
    chunks = ingest()
    embedder = get_embedder()
    vectors = embedder.embed([chunk.text for chunk in chunks])
    index = VectorIndex.build(chunks, vectors)
    tfidf = TfidfRetriever(chunks)
    _chunks, _tfidf, _index, _embedder = chunks, tfidf, index, embedder
    return chunks, tfidf, index, embedder


def _ensure() -> tuple[TfidfRetriever, VectorIndex, Embedder]:
    if _tfidf is None or _index is None or _embedder is None:
        _, tfidf, index, embedder = _rebuild()
        return tfidf, index, embedder
    return _tfidf, _index, _embedder


@app.get("/health")
def health() -> dict[str, object]:
    embedder = _embedder or get_embedder()
    return {
        "status": "ok",
        "embedder": embedder.name,
        "embeddings_abstain_threshold": EMBEDDINGS_ABSTAIN_THRESHOLD,
        "tfidf_abstain_threshold": TFIDF_ABSTAIN_THRESHOLD,
    }


@app.post("/ingest")
def ingest_corpus() -> dict[str, object]:
    chunks, _, _, embedder = _rebuild()
    return {"chunks": len(chunks), "embedder": embedder.name, "indexed": len(chunks)}


@app.get("/retrieve")
def retrieve(
    q: str = Query(..., min_length=1),
    k: int = Query(4, ge=1, le=10),
    retriever: str = Query("embeddings"),
) -> dict[str, object]:
    tfidf, index, embedder = _ensure()
    name = retriever.strip().lower()
    if name == "tfidf":
        hits = tfidf.search(q, limit=k)
        threshold = TFIDF_ABSTAIN_THRESHOLD
        used = "tfidf"
    else:
        query_vector = embedder.embed([q])[0]
        hits = index.search(query_vector, limit=k)
        threshold = EMBEDDINGS_ABSTAIN_THRESHOLD
        used = "embeddings"
    matches = [
        {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "text": chunk.text,
            "score": score,
        }
        for chunk, score in hits
    ]
    best = matches[0]["score"] if matches else 0.0
    return {
        "retriever": used,
        "embedder": embedder.name,
        "query": q,
        "abstain_threshold": threshold,
        "would_abstain": best < threshold,
        "matches": matches,
    }
