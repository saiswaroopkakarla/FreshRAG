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

        # requests defaults to ISO-8859-1 whenever the server doesn't
        # explicitly declare a charset in the Content-Type header --
        # even when the actual page is UTF-8 (extremely common: many
        # sites only declare charset via a <meta charset="utf-8"> tag,
        # not the HTTP header). Decoding UTF-8 bytes as ISO-8859-1
        # corrupts every en-dash, curly quote, and accented character
        # into "mojibake" (e.g. an en-dash becomes "\u00e2\u0080\u0093").
        # Detecting encoding from the actual content bytes instead is
        # far more reliable than trusting a possibly-absent header.
        if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding

        return resp.text
    except requests.RequestException as exc:
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc
