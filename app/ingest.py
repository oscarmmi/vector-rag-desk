from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.chunker import split_text
from app.corpus import load_documents

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks.json"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    index: int


def ingest() -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_documents():
        source = f"{document.title} {document.body}"
        for index, text in enumerate(split_text(source)):
            chunks.append(
                Chunk(
                    id=f"{document.doc_id}-{index}",
                    doc_id=document.doc_id,
                    title=document.title,
                    text=text,
                    index=index,
                )
            )
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2) + "\n",
        encoding="utf-8",
    )
    return chunks


def load_chunks() -> list[Chunk]:
    if not CHUNKS_PATH.exists():
        return ingest()
    payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in payload]
