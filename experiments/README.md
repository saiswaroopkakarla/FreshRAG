# Evaluation workflow

This turns "I need Results and Comparative Analysis sections" into a
concrete, repeatable process: run queries → label them yourself →
compute metrics automatically.

## Step 0: Start the backend

```bash
cd ..                     # project root
./run_backend.sh
```
Leave it running in one terminal for everything below.

## Step 1: Run the test queries

```bash
python experiments/run_experiment.py --tag baseline
```

This sends every query in `queries.json` (18 queries spanning
finance/sports/weather/news/tech, both time-sensitive and static) to
your running backend and writes `experiments/results_baseline.xlsx`.
Feel free to edit `queries.json` to add/remove queries first.

## Step 2: Label it

Open `results_baseline.xlsx`. Go to the **Results** sheet. Every row is
one ranked source for one query. Fill in the yellow **human_relevance**
column for every row:

| Score | Meaning |
|---|---|
| 3 | Excellent — directly answers it, current and reliable |
| 2 | Relevant but not ideal (a bit outdated, or partial) |
| 1 | Barely related |
| 0 | Irrelevant |

This is the only manual step — for 18 queries × 8 results, expect
maybe 1-2 hours. Save the file when done.

## Step 3: Compute metrics

```bash
python experiments/compute_metrics.py experiments/results_baseline.xlsx
```

Writes `experiments/metrics_summary.xlsx` with Precision@k, Recall@k
(pooled — see note below), nDCG@k, and freshness-satisfaction, both
per-query and averaged. This averaged table is your **Results**
section.

## Step 4: Repeat for each ablation, then compare

For each thing you want to compare, change the setting in `.env`,
**restart the backend**, and run Step 1 again with a different `--tag`:

```bash
# e.g. compare freshness decay functions
# .env: FRESHNESS_DECAY=linear   -> restart backend ->
python experiments/run_experiment.py --tag linear_decay

# .env: FRESHNESS_DECAY=logistic -> restart backend ->
python experiments/run_experiment.py --tag logistic_decay

# e.g. compare query understanding
# .env: QUERY_UNDERSTANDING_MODE=rule-based -> restart backend ->
python experiments/run_experiment.py --tag rule_based
```

Label each new spreadsheet the same way (Step 2), then pass **all of
them together** to get a side-by-side table:

```bash
python experiments/compute_metrics.py experiments/results_baseline.xlsx experiments/results_linear_decay.xlsx experiments/results_logistic_decay.xlsx experiments/results_rule_based.xlsx
```

The `Comparison Summary` sheet in the output is your **Comparative
Analysis** section — one row per configuration, same metrics, side by
side.

## Note on Recall@k

True recall requires knowing every relevant document that exists
anywhere, which isn't knowable for open web retrieval. What's computed
here is *pooled* recall — a standard IR evaluation practice when
exhaustive judging isn't possible: for each query, the "total relevant"
denominator is the union of everything judged relevant across **all**
the runs you feed into `compute_metrics.py` together, not the entire
web. Mention this as a stated limitation in your report — it's a
normal, defensible thing to disclose, not a flaw unique to this project.

Note also that `recall@k` is trivially 1.0 when `k` equals the total
number of results judged per query (there's nothing left to miss) —
only `recall@k` for `k` smaller than your `--top-k` is a meaningful
signal of whether relevant results got ranked early.
