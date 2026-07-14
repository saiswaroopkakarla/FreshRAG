"""Custom exceptions used across FreshRAG.

Keeping these centralized makes it easy for the FastAPI layer to catch
specific failure modes and turn them into clean HTTP error responses,
instead of leaking raw tracebacks to the client.
"""


class FreshRAGError(Exception):
    """Base class for all FreshRAG-specific errors."""


class SearchProviderError(FreshRAGError):
    """Raised when all configured search providers fail to return results."""


class FetchError(FreshRAGError):
    """Raised when a web page cannot be downloaded or parsed."""


class NoContentRetrievedError(FreshRAGError):
    """Raised when the pipeline could not gather any usable content for a query."""


class EmbeddingError(FreshRAGError):
    """Raised when embedding generation fails."""


class GenerationError(FreshRAGError):
    """Raised when the final-answer generation step fails."""
