from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


def document_ids() -> set[str]:
    return {path.stem for path in CORPUS_DIR.glob("*.md")}
