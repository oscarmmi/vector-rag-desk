# Vector RAG Desk

Python FastAPI desk that answers questions over a small field-operations handbook with **citations** and an **I don't know** path.

The handbook is the same eight documents used in [cite-rag-desk](https://github.com/oscarmmi/cite-rag-desk) so retrieval quality can be compared 1:1: **TF-IDF vs embeddings**. This repo is the embeddings side (sqlite-vec + sentence-transformers). It is not another Harbor Desk UI clone.

## Status

- Slice 1: app skeleton, corpus, 34-question gold set, `/health`.
- Slice 2: overlapping 400-char chunks and a TF-IDF cosine retriever.
- Slice 3: `all-MiniLM-L6-v2` embeddings in sqlite-vec (`data/index.sqlite`), same chunks as TF-IDF. Abstain threshold for embeddings is **0.35 cosine similarity** (TF-IDF stays 0.08, same as cite-rag-desk). The answer slice will use that cutoff.
- Later: answer + abstain, eval script that **writes** the metrics table below from `data/gold.json`. Do not type numbers into that table by hand.

## Run locally

Python 3.12+. No API key required for the default extractive path.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) — it should return `{"status":"ok"}`.

Chunk the handbook, embed it, and search:

```bash
curl -X POST http://127.0.0.1:8000/ingest
curl "http://127.0.0.1:8000/retrieve?q=What+time+does+dispatch+review+the+order+queue"
curl "http://127.0.0.1:8000/retrieve?q=What+time+does+dispatch+review+the+order+queue&retriever=tfidf"
```

Default retriever is **embeddings**. `would_abstain` is true when the best score is below the threshold (0.35 embeddings / 0.08 TF-IDF). The next slice will return "I don't know" on that path.

`VECTOR_RAG_EMBEDDER=fake` uses hashed bag-of-words vectors (pytest). Unset it (or set `minilm`) to use sentence-transformers `all-MiniLM-L6-v2`.

`data/chunks.json` is the shared chunk list. `data/index.sqlite` is the sqlite-vec index.

## Tests

```bash
pytest
```

## Eval

Gold set: `data/gold.json` (34 questions, 5 of them should abstain).

When the eval slice lands, run:

```bash
python -m app.eval
```

That command will overwrite the table in this README. Empty until then:

| retriever   | recall@4 | abstention rate | mean latency (ms) |
| ----------- | -------- | --------------- | ----------------- |
| TF-IDF      | —        | —               | —                 |
| embeddings  | —        | —               | —                 |

## Corpus

`data/corpus/*.md` — eight English handbook pages (dispatch, failed delivery, time windows, handoff, loading, POD, closeout, safety).
