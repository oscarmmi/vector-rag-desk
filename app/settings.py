from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "index.sqlite"

# Cosine similarity below this value will abstain in the answer slice.
# MiniLM scores on this handbook typically sit ~0.2–0.7; 0.35 is a weak-match cutoff.
EMBEDDINGS_ABSTAIN_THRESHOLD = 0.35

# cite-rag-desk used 0.08 on the same TF-IDF formula.
TFIDF_ABSTAIN_THRESHOLD = 0.08

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
