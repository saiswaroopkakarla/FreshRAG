"""
Module 3 -- Web Page Fetcher.

Downloads a URL and hands raw HTML off to processing.cleaner /
processing.metadata. Kept deliberately dumb (no metadata parsing here)
so this module only ever has one job: get the bytes.
"""

import logging

import requests

from utils.exceptions import FetchError

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 FreshRAG-Bot/1.0"
    )
}


def fetch_html(url: str, timeout: int = 8) -> str:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            raise FetchError(f"Non-HTML content type for {url}: {content_type}")
        return resp.text
    except requests.RequestException as exc:
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc
