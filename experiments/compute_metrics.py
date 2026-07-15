"""
Step 2 of the evaluation workflow: read one or more labeled result
spreadsheets (produced by run_experiment.py and filled in by hand) and
compute Precision@k, nDCG@k, and freshness-satisfaction, both per-query
and averaged, using the same evaluation/metrics.py functions used
elsewhere in the project.

Usage:
    # Single run:
    python experiments/compute_metrics.py experiments/results_baseline.xlsx

    # Compare multiple runs side by side (this table is your
    # Comparative Analysis section):
    python experiments/compute_metrics.py experiments/results_baseline.xlsx experiments/results_linear_decay.xlsx experiments/results_rule_based.xlsx

Output: experiments/metrics_summary.xlsx

Note on "Recall@k": true recall requires knowing the total number of
relevant documents that exist anywhere, which isn't knowable for open
web retrieval. What's reported here is *pooled* recall -- relevant
results found within the top-k you actually judged, as a fraction of
all relevant results found anywhere in your judged set for that query
(a standard practice in IR evaluation when exhaustive judging isn't
possible; document this limitation in your report).
"""

import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, freshness_satisfaction  # noqa: E402

HEADER_FILL = PatternFill(start_color="2E5395", end_color="2E5395", fill_type="solid")
HEADER_FONT = Font(name="Calibri", bold=True, color="FFFFFF")
BODY_FONT = Font(name="Calibri", size=11)
RELEVANT_THRESHOLD = 2  # human_relevance >= this counts as "relevant" for precision/recall
K_VALUES = [3, 5, 8]


def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Results")
    missing = df["human_relevance"].isna()
    if missing.any():
        n = missing.sum()
        print(f"  WARNING: {n} row(s) in {path.name} have no human_relevance filled in -- "
              f"they will be excluded from metrics for their query.", file=sys.stderr)
    df = df.dropna(subset=["human_relevance"])
    df["human_relevance"] = df["human_relevance"].astype(int)
    df["rank"] = df["rank"].astype(int)
    return df


def compute_per_query_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Pooled relevance set per query: the union of relevant URLs found
    # across ALL runs for that query. Standard IR "pooling" practice --
    # lets recall@k be meaningful (not trivially 1.0) even though we
    # can't judge the entire web, by measuring how much of what WAS
    # found (across every config tested) each individual run surfaced
    # within its own top-k.
    pooled_relevant_by_query = {
        query_id: set(group.loc[group["human_relevance"] >= RELEVANT_THRESHOLD, "source_url"])
        for query_id, group in df.groupby("query_id")
    }

    records = []
    for (run_tag, query_id), group in df.groupby(["run_tag", "query_id"]):
        group = group.sort_values("rank")
        retrieved_ids = group["source_url"].tolist()
        relevance_scores = group["human_relevance"].tolist()
        relevant_ids = set(group.loc[group["human_relevance"] >= RELEVANT_THRESHOLD, "source_url"])
        pooled_relevant_ids = pooled_relevant_by_query[query_id]

        row = {
            "run_tag": run_tag,
            "query_id": query_id,
            "query": group["query"].iloc[0],
            "category": group["category"].iloc[0],
            "n_judged": len(group),
            "n_relevant": len(relevant_ids),
        }
        for k in K_VALUES:
            if k <= len(retrieved_ids):
                row[f"precision@{k}"] = round(precision_at_k(retrieved_ids, relevant_ids, k), 3)
                row[f"recall@{k}"] = round(recall_at_k(retrieved_ids, pooled_relevant_ids, k), 3)
                row[f"ndcg@{k}"] = round(ndcg_at_k(relevance_scores, k), 3)
            else:
                row[f"precision@{k}"] = None
                row[f"recall@{k}"] = None
                row[f"ndcg@{k}"] = None
        row["freshness_satisfaction"] = round(
            freshness_satisfaction(group["freshness_score"].astype(float).tolist()), 3
        )
        row["avg_final_score"] = round(group["final_score"].astype(float).mean(), 3)
        records.append(row)
    return pd.DataFrame(records)


def compute_run_summary(per_query: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_query.columns if c.startswith(("precision@", "recall@", "ndcg@"))] + [
        "freshness_satisfaction", "avg_final_score"
    ]
    summary = per_query.groupby("run_tag")[metric_cols].mean().round(3).reset_index()
    n_queries = per_query.groupby("run_tag")["query_id"].nunique().rename("n_queries")
    summary = summary.merge(n_queries, on="run_tag")
    return summary


def write_workbook(all_raw: pd.DataFrame, per_query: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    wb = Workbook()

    def write_df(ws, df: pd.DataFrame):
        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=1):
            for c_idx, value in enumerate(row, start=1):
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                if r_idx == 1:
                    cell.fill = HEADER_FILL
                    cell.font = HEADER_FONT
                else:
                    cell.font = BODY_FONT
        for c_idx, col_name in enumerate(df.columns, start=1):
            width = 42 if col_name in ("query",) else 16
            ws.column_dimensions[get_column_letter(c_idx)].width = width
        ws.freeze_panes = "A2"

    ws_summary = wb.active
    ws_summary.title = "Comparison Summary"
    ws_summary["A1"] = "Run comparison (average across all judged queries) -- this table is your Comparative Analysis"
    ws_summary["A1"].font = Font(name="Calibri", bold=True, size=12)
    ws_summary_data_start = 3
    for r_idx, row in enumerate(dataframe_to_rows(summary, index=False, header=True), start=ws_summary_data_start):
        for c_idx, value in enumerate(row, start=1):
            cell = ws_summary.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == ws_summary_data_start:
                cell.fill = HEADER_FILL
                cell.font = HEADER_FONT
            else:
                cell.font = BODY_FONT
    for c_idx in range(1, len(summary.columns) + 1):
        ws_summary.column_dimensions[get_column_letter(c_idx)].width = 16

    ws_perquery = wb.create_sheet("Per-Query Metrics")
    write_df(ws_perquery, per_query)

    ws_raw = wb.create_sheet("Raw Judged Data")
    write_df(ws_raw, all_raw)

    wb.save(out_path)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    frames = []
    for p in paths:
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading {p} ...", file=sys.stderr)
        frames.append(load_run(p))

    all_raw = pd.concat(frames, ignore_index=True)
    if all_raw.empty:
        print("No judged rows found across the given file(s) -- fill in human_relevance first.", file=sys.stderr)
        sys.exit(1)

    per_query = compute_per_query_metrics(all_raw)
    summary = compute_run_summary(per_query)

    out_path = Path(__file__).parent / "metrics_summary.xlsx"
    write_workbook(all_raw, per_query, summary, out_path)

    print(f"\nWrote comparison summary to {out_path}", file=sys.stderr)
    print("\n" + summary.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
