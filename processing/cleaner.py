"""
Content Cleaner -- strips boilerplate (nav, ads, scripts, footers) from
raw HTML and returns the main readable text. Deliberately simple
heuristic extraction (no heavy readability library dependency) so the
project stays pip-installable in seconds.
"""

import re

from bs4 import BeautifulSoup

_JUNK_TAGS = ["script", "style", "nav", "footer", "header", "form", "aside", "noscript"]


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag_name in _JUNK_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Prefer <article> if present -- most news/blog sites wrap body copy in it.
    container = soup.find("article") or soup.find("main") or soup.body or soup

    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p.split()) > 4)

    if not text:
        # Fallback: just grab all visible text.
        text = container.get_text(" ", strip=True)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""
