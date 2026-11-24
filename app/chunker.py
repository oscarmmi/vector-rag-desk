from __future__ import annotations

CHUNK_SIZE = 400
OVERLAP = 40


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    collapsed = " ".join(text.split())
    length = len(collapsed)
    if length == 0:
        return []

    chunks: list[str] = []
    start = 0
    while start < length:
        window = collapsed[start : start + chunk_size]
        end = min(start + len(window), length)
        if end < length and len(window) == chunk_size:
            last_space = window.rfind(" ")
            if last_space != -1 and last_space > int(chunk_size * 0.6):
                end = start + last_space
        chunk = collapsed[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(start + 1, end - overlap)
    return chunks
