"""
Module 1 & 2 -- Query Understanding / Query Classification.

Design decision (from the project's research discussion): start with a
transparent, rule-based analyzer rather than a black-box NLP model. This
gives a clean, explainable baseline to report in the thesis/paper, and a
concrete target to later beat with a learned classifier if desired.

The analyzer detects:
  - domain        (finance, sports, weather, news, tech, general)
  - time_sensitive (bool)
  - intent        (reason, comparison, definition, current_status, other)
  - keywords      (naive stopword-filtered keyword extraction)

These outputs feed the Adaptive Weight Generator (ranking/weight_generator.py),
which is where the actual research novelty lives.

Note on `clean_search_query`: the raw natural-language question is NOT
always what should be sent to a search engine verbatim. E.g. "whats the
recent update on FIFA" contains "whats" (no apostrophe) right next to
"update" -- search engines can and do pattern-match that combination
onto the vastly more popular query "what's the latest WhatsApp update",
silently hijacking the results. Stripping interrogative filler words and
searching on the extracted content keywords instead avoids this class
of failure.
"""

import re
from dataclasses import dataclass, field

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "in", "on", "at", "for", "to", "of", "and", "or", "but", "with", "about",
    "what", "why", "how", "who", "when", "where", "which", "will", "did",
    "does", "do", "can", "could", "should", "would", "this", "that", "it",
    "its", "as", "by", "from", "up", "down", "into", "over", "than", "then",
    # Contractions without apostrophes (common in casual typing) -- these
    # are pure filler, but left unfiltered they can accidentally collide
    # with unrelated brand names (e.g. "whats" ~ "whatsapp").
    "whats", "hows", "wheres", "whens", "whos", "thats", "theres",
    "dont", "cant", "wont", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "shouldnt", "wouldnt", "couldnt",
    "im", "youre", "theyre", "weve", "ive",
    "any", "some", "there", "here", "recent", "recently",
}

_TIME_SENSITIVE_PATTERNS = [
    r"\btoday\b", r"\bnow\b", r"\bcurrent(ly)?\b", r"\blatest\b",
    r"\bthis (week|month|year|morning|hour)\b", r"\brecent(ly)?\b",
    r"\bbreaking\b", r"\blive\b", r"\bright now\b", r"\bas of\b",
    r"\byesterday\b", r"\btonight\b", r"\bupdate[ds]?\b",
]

_DOMAIN_KEYWORDS = {
    "finance": [
        "stock", "stocks", "share", "shares", "market", "nasdaq", "nyse",
        "crypto", "bitcoin", "ethereum", "price", "earnings", "ipo",
        "inflation", "interest rate", "fed", "economy", "trading",
    ],
    "sports": [
        "match", "score", "game", "tournament", "championship", "league",
        "cup", "player", "team", "goal", "innings", "wicket", "olympics",
        "fifa", "uefa", "world cup", "premier league", "nba", "nfl", "nhl",
        "mlb", "ipl", "cricket", "football", "soccer", "tennis", "wimbledon",
        "formula 1", "f1", "rugby", "hockey", "basketball", "athlete",
    ],
    "weather": [
        "weather", "temperature", "rain", "forecast", "storm", "cyclone",
        "humidity", "climate today",
    ],
    "news": [
        "news", "election", "government", "policy", "president", "minister",
        "war", "protest", "law", "bill passed",
    ],
    "tech": [
        "software", "version", "ai model", "chip", "processor",
        "iphone", "android app", "framework", "github", "codebase",
        "api", "sdk", "llm", "chatbot", "app store", "operating system",
    ],
}

_INTENT_PATTERNS = {
    "reason": [r"\bwhy\b"],
    "comparison": [r"\bvs\b", r"\bversus\b", r"\bcompar(e|ison)\b", r"\bbetter than\b"],
    "definition": [r"\bwhat is\b", r"\bdefine\b", r"\bmeaning of\b"],
    "current_status": [r"\bwhat('|)s happening\b", r"\bstatus of\b", r"\bhow is\b"],
}


@dataclass
class QueryAnalysis:
    query: str
    domain: str = "general"
    time_sensitive: bool = False
    intent: str = "other"
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "domain": self.domain,
            "time_sensitive": self.time_sensitive,
            "intent": self.intent,
            "keywords": self.keywords,
        }


def _detect_domain(query_lower: str) -> str:
    scores = {domain: 0 for domain in _DOMAIN_KEYWORDS}
    for domain, kws in _DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in query_lower:
                scores[domain] += 1
    best_domain = max(scores, key=scores.get)
    return best_domain if scores[best_domain] > 0 else "general"


def _detect_time_sensitive(query_lower: str) -> bool:
    return any(re.search(p, query_lower) for p in _TIME_SENSITIVE_PATTERNS)


def _detect_intent(query_lower: str) -> str:
    for intent, patterns in _INTENT_PATTERNS.items():
        if any(re.search(p, query_lower) for p in patterns):
            return intent
    return "other"


def _extract_keywords(query: str, max_keywords: int = 8) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9']+", query.lower())
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:max_keywords]


def analyze_query(query: str) -> QueryAnalysis:
    """Run the full rule-based query understanding pipeline."""
    query_lower = query.lower().strip()
    return QueryAnalysis(
        query=query,
        domain=_detect_domain(query_lower),
        time_sensitive=_detect_time_sensitive(query_lower),
        intent=_detect_intent(query_lower),
        keywords=_extract_keywords(query),
    )


def clean_search_query(analysis: QueryAnalysis, min_keywords: int = 2) -> str:
    """
    Builds the actual string sent to search providers. Prefers the
    extracted content keywords (filler/interrogative words stripped) over
    the raw question, specifically to avoid the search engine
    pattern-matching onto an unrelated popular query (see module
    docstring). Falls back to the raw query if too few keywords survive
    filtering, so short/unusual queries still get *something* sent.
    """
    if len(analysis.keywords) >= min_keywords:
        return " ".join(analysis.keywords)
    return analysis.query
