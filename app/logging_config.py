"""Logging setup for FreshRAG.

Every pipeline module logs through `logging.getLogger(__name__)`. This
function configures a single consistent formatter/handler so logs from
retrieval, ranking, and generation are all readable together -- this
matters a lot once you start debugging *why* a particular document
ranked where it did.
"""

import logging
import sys

from app.config import get_settings

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet down noisy third-party loggers.
    for noisy in ("urllib3", "httpx", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
