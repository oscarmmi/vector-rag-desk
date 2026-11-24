from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str


def document_ids() -> set[str]:
    return {path.stem for path in CORPUS_DIR.glob("*.md")}


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        body = "\n".join(lines[1:]).strip()
        documents.append(Document(doc_id=path.stem, title=title, body=body))
    return documents
