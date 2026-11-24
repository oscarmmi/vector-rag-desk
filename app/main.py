from fastapi import FastAPI

app = FastAPI(
    title="Vector RAG Desk",
    summary="Embeddings RAG with citations, abstention, and a TF-IDF baseline on the same gold set.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
