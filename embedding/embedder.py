"""
Module 5 -- Embedding.

Two backends are supported:

  - "tfidf" (default): scikit-learn TF-IDF + cosine similarity. Zero
    downloads, instant startup, no GPU/torch dependency. Good enough
    baseline for keyword-heavy queries (stock tickers, names, events)
    which is most of what time-sensitive queries look like anyway.

  - "sentence-transformers": true dense semantic embeddings via a local
    transformer model (all-MiniLM-L6-v2). Better for paraphrase-style
    matching. Requires `pip install sentence-transformers` (commented
    out in requirements.txt by default because it pulls in torch).

Both backends expose the same interface so the rest of the pipeline
(ranking/relevance.py) doesn't need to know which one is active.
"""

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import get_settings
from utils.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

_st_model = None  # lazy-loaded sentence-transformers model, if used


def _get_sentence_transformer():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "EMBEDDING_MODE=sentence-transformers but the package isn't "
                "installed. Run: pip install sentence-transformers"
            ) from exc
        logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2' ...")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def compute_relevance_scores(query: str, documents: list[str]) -> list[float]:
    """
    Returns a cosine-similarity relevance score in [0, 1] for the query
    against each document/chunk in `documents`, in the same order.
    """
    if not documents:
        return []

    settings = get_settings()

    if settings.embedding_mode == "sentence-transformers":
        try:
            model = _get_sentence_transformer()
            embeddings = model.encode([query] + documents, normalize_embeddings=True)
            query_vec = embeddings[0:1]
            doc_vecs = embeddings[1:]
            sims = cosine_similarity(query_vec, doc_vecs)[0]
            return [float(max(0.0, min(1.0, s))) for s in sims]
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001 -- fall back gracefully
            logger.warning("sentence-transformers failed (%s); falling back to TF-IDF.", exc)

    # TF-IDF fallback / default path.
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([query] + documents)
        sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        return [float(max(0.0, min(1.0, s))) for s in sims]
    except ValueError:
        # Can happen if all documents are empty / pure stopwords.
        return [0.0 for _ in documents]
