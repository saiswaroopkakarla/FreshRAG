"""
Module 4 -- Chunking.

Note from the project's design discussion: since FreshRAG fetches fresh
content per-query rather than maintaining a persistent local knowledge
base, chunks are only ever held in memory for the duration of a single
request (see embedding/vector_store.py). Chunking is still needed
because a single article can be too long to embed/score as one unit.
"""

from dataclasses import dataclass

from app.config import get_settings


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source_url: str
    source_title: str
    published_date: object  # datetime | None
    author: str
    domain: str


def chunk_text(text: str, chunk_size_words: int | None = None, overlap_words: int | None = None) -> list[str]:
    settings = get_settings()
    chunk_size_words = chunk_size_words or settings.chunk_size_words
    overlap_words = overlap_words if overlap_words is not None else settings.chunk_overlap_words

    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size_words:
        return [text]

    chunks = []
    step = max(chunk_size_words - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def build_chunks_for_document(
    text: str,
    source_url: str,
    source_title: str,
    published_date,
    author: str,
    domain: str,
) -> list[Chunk]:
    raw_chunks = chunk_text(text)
    return [
        Chunk(
            text=raw,
            chunk_index=i,
            source_url=source_url,
            source_title=source_title,
            published_date=published_date,
            author=author,
            domain=domain,
        )
        for i, raw in enumerate(raw_chunks)
    ]
