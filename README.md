# FreshRAG
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Anthropic%20%7C%20OpenAI%20%7C%20Groq%20%7C%20DeepSeek-412991)


**Adaptive Multi-Source Temporal-Aware Hybrid Retrieval-Augmented Generation (AMT-RAG)**

Standard RAG systems rank retrieved documents purely by *semantic
similarity*. That fails for time-sensitive questions: ask "Why is Apple
stock crashing today?" and a normal RAG happily hands you a very similar
but 3-year-old article about Apple earnings. FreshRAG ranks sources by a
**hybrid, adaptively-weighted score** — semantic relevance, freshness,
authority, and credibility — so time-sensitive queries actually get
time-relevant answers.

Runs completely free out of the box: DuckDuckGo web search, TF-IDF
embeddings, and an extractive answer generator all need **zero API
keys**. Add keys later to upgrade any single component.

---

## Architecture

```
Query  ─▶  Query Understanding  ─▶  Adaptive Weight Generator
                 │                          │
                 ▼                          │
        Multi-Source Web Search             │
                 │                          │
                 ▼                          │
     Fetch → Clean → Metadata → Chunk       │
                 │                          │
                 ▼                          │
   Score: Semantic | Freshness | Authority | Credibility
                 │                          │
                 └──────────▶  Hybrid Ranker ◀┘
                                    │
                                    ▼
                          Top-K Ranked Sources
                                    │
                                    ▼
                           Answer Generation
```

### Project layout

```
freshrag/
├── app/
│   ├── main.py            # FastAPI app + routes
│   ├── pipeline.py         # orchestrates the full 12-module flow
│   ├── config.py            # all settings, loaded from .env
│   └── logging_config.py
├── processing/
│   ├── llm_understanding.py  # ★ Module 1/2 (primary): LLM-based query understanding
│   ├── query_analyzer.py    # Module 1/2 (fallback): rule-based domain/time-sensitivity
│   ├── cleaner.py            # strips boilerplate HTML -> main text
│   ├── metadata.py            # published date / author / domain extraction (incl. JSON-LD)
│   └── chunker.py              # sliding-window chunking
├── retriever/
│   ├── search_api.py         # DuckDuckGo (free) + optional NewsAPI/Tavily
│   └── web_fetcher.py         # HTTP page download
├── embedding/
│   ├── embedder.py            # TF-IDF (default) or sentence-transformers
│   └── vector_store.py         # in-memory, per-request chunk store
├── ranking/
│   ├── weight_generator.py    # ★ the research novelty: adaptive weights
│   ├── relevance.py
│   ├── freshness.py            # 4 decay functions: linear/exp/logistic/piecewise
│   ├── authority.py
│   ├── credibility.py
│   └── hybrid_rank.py
├── generator/
│   └── llm.py                  # Anthropic / OpenAI / Groq / DeepSeek / extractive fallback
├── evaluation/
│   └── metrics.py               # Precision@k, Recall@k, nDCG@k, freshness satisfaction
├── streamlit_app.py            # demo UI (calls the FastAPI backend)
├── requirements.txt
└── .env.example
```

---

## Quickstart

**Requirements:** Python 3.10+

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env template (works with zero keys filled in)
cp .env.example .env

# 4. Start the backend (Terminal 1)
./run_backend.sh
# or: uvicorn app.main:app --reload

# 5. Start the demo UI (Terminal 2)
./run_frontend.sh
# or: streamlit run streamlit_app.py
```

Then open:
- **Streamlit demo:** http://localhost:8501
- **API docs (Swagger):** http://localhost:8000/docs

Try asking: *"Why is Apple stock falling today?"* or *"Latest news on the interest rate decision"*.

---

## How the ranking actually works

1. **Query Understanding** (`processing/llm_understanding.py`, falling
   back to `processing/query_analyzer.py`) — **tries an LLM first**
   (Groq → DeepSeek → OpenAI → Anthropic, whichever key is configured)
   to determine time-sensitivity, intent, topic, keywords, a cleaned
   search query, and **the ranking weights directly** — reasoned about
   per-query rather than looked up from a fixed table. This is
   deliberately not restricted to a handful of domain buckets, and
   handles typos and unseen topics the way a human reading the query
   would. If no key is configured (or the call fails for any reason),
   it falls back automatically to a rule-based keyword-table analyzer
   — the system always produces a usable result, it just gets smarter
   with an LLM key. See "Why LLM-first" below for the reasoning.

2. **Multi-source retrieval** — DuckDuckGo always; NewsAPI is added for
   time-sensitive finance/news/sports queries if `NEWSAPI_KEY` is set;
   Tavily adds extra recall if `TAVILY_API_KEY` is set. Results are
   de-duplicated by URL.

3. **Fetch → Clean → Metadata → Chunk** — downloads each page, strips
   nav/ads/scripts, extracts `published_date`/`author`/`domain`
   (checking JSON-LD structured data, `<meta>` tags, and `<time>` tags
   in that order), and splits long articles into overlapping
   word-window chunks.

4. **Scoring** — each chunk gets 4 independent scores in [0, 1]:
   - **Semantic**: cosine similarity (TF-IDF by default; swap to
     `sentence-transformers` in `.env` for true dense embeddings)
   - **Freshness**: pick a decay function in `.env` — linear,
     exponential (default), logistic, or piecewise — and compare them
     as an ablation study
   - **Authority**: curated reputable-domain tiers + `.gov`/`.edu` boosts
   - **Credibility**: presence of author, publish date, HTTPS

5. **Hybrid Ranker** — `final = w_sem·semantic + w_fresh·freshness + w_auth·authority + w_cred·credibility`,
   using the weights from step 1.

6. **Answer Generation** — tried in priority order: Anthropic → OpenAI →
   Groq → DeepSeek → built-in extractive fallback. Any one key is
   enough; Groq/DeepSeek mean you can get a real synthesized, cited
   answer with zero cost, not just the extractive summary. Setting a
   key here reuses the same key from Stage 1 automatically — no double
   configuration needed. With no key at all, a deterministic extractive
   summary of the top sources is returned instead — the app never
   breaks due to a missing key.

### Why LLM-first for query understanding

A fixed keyword table has a hard ceiling: it can only recognize domains
and phrasings it was explicitly told about in advance, and it can
misfire in hard-to-predict ways (e.g. a generic word like `"update"`
being tied to one domain can misclassify an unrelated query, and
raw/unfiltered query text sent to a search engine can collide with an
unrelated but far more popular query). An LLM has no such ceiling: it
reasons about arbitrary topics and typos using general knowledge, and
proposes the ranking weights directly per query instead of mapping to
one of a handful of pre-defined domains.

The rule-based analyzer still ships and still runs automatically
whenever no LLM key is configured, so the project's zero-cost,
zero-key guarantee is unchanged — you just get a materially better
Stage 1 the moment you add a free Groq key.



---

## Configuring things (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` / `DEEPSEEK_API_KEY` | empty | Optional: powers query understanding (Stage 1) AND answer generation (Stage 11) via LLM instead of the rule-based/extractive fallbacks. Groq recommended (genuinely free tier) |
| `QUERY_UNDERSTANDING_MODE` | `auto` | `auto` (LLM if key present, else rule-based) \| `llm` (force, errors without a key) \| `rule-based` (force, for ablation) |
| `EMBEDDING_MODE` | `tfidf` | `tfidf` (free/instant) or `sentence-transformers` (semantic, needs extra install) |
| `FRESHNESS_DECAY` | `exponential` | `linear` \| `exponential` \| `logistic` \| `piecewise` |
| `FRESHNESS_LAMBDA_DAYS` | `3.0` | Decay speed for exponential |
| `TOP_K_RESULTS` | `8` | How many ranked sources to return |
| `MAX_URLS_TO_FETCH` | `10` | How many search results to attempt fetching |
| `NEWSAPI_KEY` / `TAVILY_API_KEY` | empty | Optional extra search providers |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | empty | Optional LLM-generated (vs extractive) final answers, AND usable as a query-understanding provider too |

To use real semantic embeddings instead of TF-IDF:
```bash
pip install sentence-transformers
# then in .env:
EMBEDDING_MODE=sentence-transformers
```

---

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/config` | GET | Shows active configuration + which optional keys are set |
| `/analyze` | POST `{"query": "..."}` | Runs only query understanding + weight generation (no retrieval) — useful for debugging domain/time-sensitivity detection |
| `/query` | POST `{"query": "...", "top_k": 8}` | Full pipeline: retrieve → score → rank → generate |

---

## Notes & limitations

- Free web search (DuckDuckGo via `ddgs`) can occasionally rate-limit;
  add `TAVILY_API_KEY` or `NEWSAPI_KEY` for more reliable volume.
- Metadata extraction (published date/author) depends on how well a
  given site tags its HTML; pages with no discoverable date get a
  neutral (0.5) freshness score rather than being penalized.
- This is a research prototype, not a production scraper — respect
  target sites' `robots.txt` and rate limits if you scale usage up.
