# FreshRAG

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Anthropic%20%7C%20OpenAI%20%7C%20Groq%20%7C%20DeepSeek-412991)

**Adaptive Multi-Source Temporal-Aware Hybrid RAG (AMT-RAG)**

Standard RAG systems rank retrieved documents purely by *semantic similarity*. That fails for time-sensitive queries — ask *"Why is Apple stock crashing today?"* and a normal RAG happily hands you a semantically-similar-but-3-year-old earnings article. FreshRAG fixes this by ranking sources with a **hybrid, adaptively-weighted score** across four dimensions: semantic relevance, freshness, authority, and credibility — so time-sensitive queries get time-relevant answers.

Runs completely **free out of the box**: DuckDuckGo search, TF-IDF embeddings, and an extractive answer generator need zero API keys. Add keys to upgrade any single component.

---

## Architecture — 12-Module Pipeline

```
Query
  │
  ▼
[Module 1/2] Query Understanding
  ├── Primary: LLM-based (Groq → DeepSeek → OpenAI → Anthropic)
  │   Tolerates typos, open-domain queries, novel topics
  └── Fallback: Rule-based keyword analyzer — works with zero API keys
  │
  ▼
[Module 3] Adaptive Weight Generator   ◀── Main research contribution
  │  Adapts ranking weights per query:
  │  "Apple stock crashing today"   → freshness-heavy (0.50 fresh, 0.30 semantic)
  │  "What is Pythagorean theorem?" → semantic-heavy  (0.70 semantic, 0.05 fresh)
  │
  ▼
[Module 4] Multi-Source Web Search
  ├── DuckDuckGo (free, no key — with retry + backend fallback)
  ├── NewsAPI    (optional)
  └── Tavily     (optional)
  │
  ▼
[Module 5] Web Fetch
[Module 6] Metadata Extraction (author, domain, publish date)
[Module 7] Chunking
  │
  ▼
[Module 8]  Semantic Relevance  — TF-IDF cosine or sentence-transformers
[Module 9]  Freshness Score     — 4 decay functions: linear / exponential / logistic / piecewise
[Module 10] Authority Score     — tiered domain list (reuters.com, bbc.com ...) + .gov/.edu/.org boosts
[Module 11] Credibility Score   — author present (+0.25) + publish date present (+0.25) + HTTPS (+0.10)
  │
  ▼
[Module 12] Hybrid Ranker
  final = w_sem·semantic + w_fresh·freshness + w_auth·authority + w_cred·credibility
  │
  ▼
Answer Generation
  ├── LLM synthesis (Anthropic → OpenAI → Groq → DeepSeek, tried in order)
  └── Extractive fallback — no API key needed, always works
```

---

## Key Design Decisions

**Why adaptive weights?**
Fixed-weight formulas (`0.45·semantic + 0.30·freshness + ...`) can't serve both time-sensitive and timeless queries well. The weight generator reasons about each specific query to decide how much freshness vs semantic relevance matters — this is the project's core research contribution.

**Why four freshness decay functions?**
Different use cases have different "staleness curves." The four functions are selectable via `.env` for ablation studies — a clean way to compare them in a paper or report.

**Why pluggable LLM providers?**
No lock-in. Query understanding and answer generation each independently try their own provider chain. Set whichever key you have; the system uses it automatically and falls back gracefully to free options.

**Why DuckDuckGo as default?**
Zero cost, no signup, works immediately. Upgrading to NewsAPI or Tavily is a one-line `.env` change.

---

## Freshness Scoring — 4 Decay Functions

All functions take `age_days ≥ 0` and return a score in `[0, 1]`.

| Function | Formula | When to use |
|---|---|---|
| Linear | `F = max(0, 1 - age/max_age)` | Gradual staleness (general news) |
| Exponential | `F = e^(-λ·age)` | Fast-decay (stock prices, sports scores) |
| Logistic | `F = 1 / (1 + e^(k·(age-midpoint)))` | Sharp "cliff" (election results) |
| Piecewise | Flat near 1.0, then exponential drop | Mixed freshness requirements |

Documents with no extractable publish date get a **neutral score of 0.5** — not penalised for missing `<meta>` tags.

---

## Credibility Scoring

Lightweight, explainable heuristic — distinct from Authority (which asks "how reputable is this publisher?"). Credibility asks "does *this specific page* show the transparency signals of trustworthy content?":

| Signal | Score contribution |
|---|---|
| Baseline | +0.40 |
| Author name present | +0.25 |
| Publish date present | +0.25 |
| HTTPS | +0.10 |

---

## Authority Scoring

Tiered domain reputation lookup:

| Tier | Examples | Score |
|---|---|---|
| Tier 1 | reuters.com, bloomberg.com, bbc.com, nature.com, who.int | 1.00 |
| Tier 2 | cnn.com, forbes.com, techcrunch.com, wired.com | 0.75 |
| .gov / .gov.in | — | 0.95 |
| .edu | — | 0.85 |
| .org | — | 0.60 |
| Unknown | — | 0.45 |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/config` | Active configuration (keys, modes, settings) |
| `POST` | `/analyze` | Stage 1 only — classify query + generate weights without retrieval |
| `POST` | `/query` | Full pipeline — retrieve, rank, generate answer |

Full Swagger docs: `http://localhost:8000/docs`

---

## Evaluation Workflow

`experiments/` contains the tooling to produce Results and Comparative Analysis — it does **not** yet contain any completed evaluation. Currently in that folder: `run_experiment.py` and `compute_metrics.py` (both tested end-to-end against a live backend), plus a starter set of 18 test queries. Running the actual evaluation is a manual next step: run the queries, label the results yourself in the generated spreadsheet (relevance judgments — no dataset provides this for open web retrieval), then compute Precision@k, Recall@k, nDCG@k, and freshness-satisfaction automatically, including a side-by-side comparison across configurations. Full walkthrough in `experiments/README.md`.

---

## Repository Structure

```
FreshRAG/
├── app/
│   ├── main.py                 # FastAPI app + routes
│   ├── pipeline.py             # 12-module pipeline orchestrator
│   ├── config.py               # All settings via .env (pydantic-settings)
│   └── logging_config.py
├── processing/
│   ├── llm_understanding.py    # Module 1/2 primary: LLM query understanding
│   ├── query_analyzer.py       # Module 1/2 fallback: rule-based classifier
│   ├── cleaner.py              # HTML → clean text
│   ├── metadata.py             # Module 6: author, domain, publish date extraction
│   └── chunker.py              # Module 7: document chunking
├── ranking/
│   ├── weight_generator.py     # ★ Module 3: adaptive weight generation
│   ├── relevance.py            # Module 8: semantic relevance (wraps embedder)
│   ├── freshness.py            # Module 9: 4 decay functions
│   ├── authority.py            # Module 10: tiered domain authority scoring
│   ├── credibility.py          # Module 11: author + date + HTTPS signals
│   └── hybrid_rank.py          # Module 12: final score + sort
├── retriever/
│   ├── search_api.py           # Module 4: DuckDuckGo + NewsAPI + Tavily
│   └── web_fetcher.py          # Module 5: fetch + parse web pages
├── embedding/
│   ├── embedder.py             # TF-IDF or sentence-transformers
│   └── vector_store.py         # In-session chunk store + ScoredChunk dataclass
├── generator/
│   └── llm.py                  # Answer generation (4 providers + extractive fallback)
├── evaluation/
│   └── metrics.py              # Precision@k, Recall@k, nDCG@k, freshness_satisfaction
├── experiments/
│   ├── run_experiment.py       # Runs test queries -> labeling spreadsheet
│   ├── compute_metrics.py      # Labeled spreadsheet -> metrics + comparison table
│   ├── queries.json            # Starter set of 18 test queries
│   └── README.md               # Full evaluation workflow walkthrough
├── utils/
│   └── exceptions.py
├── streamlit_app.py            # Demo UI — thin client calling FastAPI backend
├── run_backend.sh
├── run_frontend.sh
├── .env.example
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/saiswaroopkakarla/FreshRAG.git
cd FreshRAG
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# All API keys are optional — system works with none set
```

---

## Run

```bash
# Terminal 1 — FastAPI backend
bash run_backend.sh
# or: uvicorn app.main:app --reload

# Terminal 2 — Streamlit UI
bash run_frontend.sh
# or: streamlit run streamlit_app.py
```

Open `http://localhost:8501` for the UI · `http://localhost:8000/docs` for Swagger.

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `QUERY_UNDERSTANDING_MODE` | `auto` | `auto` / `llm` / `rule-based` |
| `EMBEDDING_MODE` | `tfidf` | `tfidf` / `sentence-transformers` |
| `FRESHNESS_DECAY` | `exponential` | `linear` / `exponential` / `logistic` / `piecewise` |
| `GROQ_API_KEY` | — | Free tier, fast |
| `DEEPSEEK_API_KEY` | — | Cheap, OpenAI-compatible |
| `OPENAI_API_KEY` | — | Optional |
| `ANTHROPIC_API_KEY` | — | Optional |
| `NEWSAPI_KEY` | — | More reliable than DDG |
| `TAVILY_API_KEY` | — | Web search API |

---

## Tech Stack

| Component | Technology |
|---|---|
| API backend | FastAPI + Uvicorn |
| Demo UI | Streamlit |
| Search | DuckDuckGo (ddgs) · NewsAPI · Tavily |
| Embeddings | TF-IDF (scikit-learn) · sentence-transformers (optional) |
| LLM providers | Anthropic · OpenAI · Groq · DeepSeek |
| Config | pydantic-settings + .env |
| Web parsing | BeautifulSoup4 + lxml |

---

## Notes & Limitations

- Free web search (DuckDuckGo via `ddgs`) can occasionally rate-limit; add `TAVILY_API_KEY` or `NEWSAPI_KEY` for more reliable volume.
- Metadata extraction (published date/author) depends on how well a given site tags its HTML; pages with no discoverable date get a neutral (0.5) freshness score rather than being penalized.
- This is a research prototype, not a production scraper — respect target sites' `robots.txt` and rate limits if you scale usage up.

---

## Author

**Kakarla Sai Swaroop** — M25DE1023, IIT Jodhpur M.Tech Data Engineering
