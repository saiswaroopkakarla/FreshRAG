"""
In-memory chunk store.

Design decision (from the project's research discussion): since FreshRAG
always fetches fresh content per query rather than maintaining a static
knowledge base, there's no need for a persistent vector database (FAISS/
Pinecone/etc). This simple in-memory store just holds chunks + their
computed scores for the duration of a single request/response cycle.

If you later want to *also* blend in a curated local knowledge base
(discussed as a possible extension), this is the class to swap out for
a real persistent store -- the interface is intentionally minimal.
"""

from dataclasses import dataclass, field

from processing.chunker import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    semantic_score: float = 0.0
    freshness_score: float = 0.0
    authority_score: float = 0.0
    credibility_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.chunk.text,
            "source_url": self.chunk.source_url,
            "source_title": self.chunk.source_title,
            "published_date": (
                self.chunk.published_date.isoformat() if self.chunk.published_date else None
            ),
            "author": self.chunk.author,
            "domain": self.chunk.domain,
            "scores": {
                "semantic": round(self.semantic_score, 4),
                "freshness": round(self.freshness_score, 4),
                "authority": round(self.authority_score, 4),
                "credibility": round(self.credibility_score, 4),
                "final": round(self.final_score, 4),
            },
        }


@dataclass
class SessionStore:
    """Holds all chunks retrieved+scored for a single query."""

    chunks: list[ScoredChunk] = field(default_factory=list)

    def add(self, scored_chunk: ScoredChunk) -> None:
        self.chunks.append(scored_chunk)

    def top_k(self, k: int) -> list[ScoredChunk]:
        return sorted(self.chunks, key=lambda c: c.final_score, reverse=True)[:k]
