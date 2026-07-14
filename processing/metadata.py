"""
Module 5 -- Metadata Extraction.

Pulls the signals the ranking stage needs from a page:
  - published_date  (feeds Freshness Scorer)
  - author          (feeds Credibility Scorer)
  - site_name / domain (feeds Authority Scorer)

Real-world pages are inconsistent about where they put the publish
date, so this tries several common locations before giving up (in
which case freshness scoring falls back to a neutral/unknown score
rather than crashing the pipeline). In rough priority order:

  1. JSON-LD structured data (schema.org Article/NewsArticle) -- this
     is the most reliable source when present, and very common on
     news/finance sites (finviz, financialcontent, etc. all use it).
  2. <meta> tags (Open Graph, Dublin Core, common CMS variants).
  3. <time> element.
  4. A raw YYYY-MM-DD-ish pattern in the visible text, as a last resort.
"""

import json
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from dateutil import parser as dateparser

logger = logging.getLogger(__name__)

_DATE_META_NAMES = [
    ("meta", {"property": "article:published_time"}),
    ("meta", {"name": "article:published_time"}),
    ("meta", {"property": "og:published_time"}),
    ("meta", {"property": "og:updated_time"}),
    ("meta", {"name": "publish-date"}),
    ("meta", {"name": "publishdate"}),
    ("meta", {"name": "date"}),
    ("meta", {"name": "dc.date.issued"}),
    ("meta", {"name": "dcterms.date"}),
    ("meta", {"itemprop": "datePublished"}),
    ("meta", {"name": "sailthru.date"}),
    ("meta", {"name": "parsely-pub-date"}),
    ("meta", {"name": "cxenseparse:recs:publishtime"}),
    ("meta", {"name": "last-modified"}),
]

# JSON-LD key names that commonly hold the publish date on
# schema.org/Article, NewsArticle, and BlogPosting objects.
_JSONLD_DATE_KEYS = ["datePublished", "dateCreated", "dateModified", "uploadDate"]

_DATE_TEXT_PATTERN = re.compile(
    r"\b(20\d{2}|19\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"
)


def _try_parse(value: str):
    try:
        dt = dateparser.parse(value, fuzzy=True)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Reject obviously-bogus future dates or pre-web dates.
        now = datetime.now(timezone.utc)
        if dt > now or dt.year < 1995:
            return None
        return dt
    except (ValueError, OverflowError):
        return None


def _find_date_in_jsonld_obj(obj) -> str | None:
    """Recursively search a parsed JSON-LD object (dict, list, or nested
    combination of both -- schema.org allows @graph arrays, single
    objects, or arrays of objects) for a usable date field."""
    if isinstance(obj, dict):
        for key in _JSONLD_DATE_KEYS:
            if key in obj and isinstance(obj[key], str):
                return obj[key]
        # schema.org @graph wrapping, or nested "mainEntity" etc.
        for value in obj.values():
            found = _find_date_in_jsonld_obj(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_date_in_jsonld_obj(item)
            if found:
                return found
    return None


def _extract_from_jsonld(soup: BeautifulSoup):
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        date_str = _find_date_in_jsonld_obj(data)
        if date_str:
            dt = _try_parse(date_str)
            if dt:
                return dt
    return None


def extract_published_date(html: str):
    soup = BeautifulSoup(html, "lxml")

    # 1. JSON-LD structured data (most reliable when present).
    dt = _extract_from_jsonld(soup)
    if dt:
        return dt

    # 2. <meta> tags.
    for tag_name, attrs in _DATE_META_NAMES:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get("content"):
            dt = _try_parse(tag["content"])
            if dt:
                return dt

    # 3. <time> element.
    time_tag = soup.find("time")
    if time_tag:
        candidate = time_tag.get("datetime") or time_tag.get_text(strip=True)
        dt = _try_parse(candidate)
        if dt:
            return dt

    # 4. Last resort: scan visible text for a YYYY-MM-DD-ish pattern.
    text_sample = soup.get_text(" ", strip=True)[:3000]
    match = _DATE_TEXT_PATTERN.search(text_sample)
    if match:
        dt = _try_parse(match.group(0))
        if dt:
            return dt

    return None


def extract_author(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for attrs in ({"name": "author"}, {"property": "article:author"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return tag["content"].strip()

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        author = _find_author_in_jsonld_obj(data)
        if author:
            return author[:120]

    byline = soup.find(class_=re.compile("byline|author", re.I))
    if byline:
        return byline.get_text(strip=True)[:120]

    return ""


def _find_author_in_jsonld_obj(obj) -> str | None:
    if isinstance(obj, dict):
        if "author" in obj:
            author = obj["author"]
            if isinstance(author, str):
                return author
            if isinstance(author, dict) and isinstance(author.get("name"), str):
                return author["name"]
            if isinstance(author, list) and author:
                first = author[0]
                if isinstance(first, dict) and isinstance(first.get("name"), str):
                    return first["name"]
                if isinstance(first, str):
                    return first
        for value in obj.values():
            found = _find_author_in_jsonld_obj(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_author_in_jsonld_obj(item)
            if found:
                return found
    return None


def extract_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc.replace("www.", "")
    except Exception:  # noqa: BLE001
        return ""
