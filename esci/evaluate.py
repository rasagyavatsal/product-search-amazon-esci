"""Run every ranker on the test queries and write the comparison table + PR curves.

Usage: uv run python -m esci.evaluate [--max-queries N]

Setup: full-catalog retrieval. Each method retrieves the top 100 products for
each query from the whole US catalog; products with no judgment for the query
count as non-relevant (so absolute numbers are a lower bound).
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from esci.data import ARTIFACTS_DIR, RESULTS_DIR, load_eval_set
from esci.fusion import rrf
from esci.metrics import RECALL_LEVELS, evaluate_run
from esci.rankers import BM25Ranker, DenseRanker, TfidfRanker

K = 100
SETTINGS = {
    "strict": frozenset("E"),        # relevant = Exact
    "lenient": frozenset("ES"),      # relevant = Exact or Substitute
}
HYBRIDS = {
    "RRF (BM25 + ST)": ["BM25", "SentenceTransformer"],
    "RRF (TF-IDF + BM25 + ST)": ["TF-IDF", "BM25", "SentenceTransformer"],
}


def run_rankers(queries: dict[str, str]) -> tuple[dict, dict]:
    qids, texts = list(queries), list(queries.values())
    runs, latency = {}, {}
    for cls, sub in [(TfidfRanker, "tfidf"), (BM25Ranker, "bm25"), (DenseRanker, "dense")]:
        path = ARTIFACTS_DIR / sub
        if not path.exists():
            print(f"skipping {cls.name}: no index at {path}")
            continue
        ranker = cls.load(path)
        t = time.perf_counter()
        hits = ranker.search(texts, k=K)
        latency[cls.name] = (time.perf_counter() - t) / len(texts) * 1000
        runs[cls.name] = {q: [pid for pid, _ in h] for q, h in zip(qids, hits)}
        print(f"{cls.name}: {latency[cls.name]:.2f} ms/query", flush=True)
        del ranker

    for name, parts in HYBRIDS.items():
        if not all(p in runs for p in parts):
            continue
        t = time.perf_counter()
        runs[name] = {q: rrf([runs[p][q] for p in parts], k=60, top_n=K) for q in qids}
        fusion_ms = (time.perf_counter() - t) / len(qids) * 1000
        latency[name] = sum(latency[p] for p in parts) + fusion_ms
    return runs, latency


def write_results(runs, latency, qrels):
    RESULTS_DIR.mkdir(exist_ok=True)
    tables, curves = {}, {}
    for setting, rel in SETTINGS.items():
        rows = []
        for name, run in runs.items():
            m = evaluate_run(run, qrels, rel_labels=rel, p_k=10, r_k=K, ndcg_k=10)
            curves[(setting, name)] = m.pop("pr_curve")
            n = m.pop("n_queries")
            rows.append({"Method": name, **m, "Latency (ms/query)": latency[name]})
        df = pd.DataFrame(rows).set_index("Method")
        df.to_csv(RESULTS_DIR / f"metrics_{setting}.csv", float_format="%.4f")
        tables[setting] = (df, n)

    with open(RESULTS_DIR / "metrics.md", "w") as f:
        for setting, (df, n) in tables.items():
            rel = " or ".join(sorted(SETTINGS[setting]))
            f.write(f"## {setting.capitalize()} relevance ({rel} = relevant), {n:,} queries\n\n")
            f.write(df.to_markdown(floatfmt=".4f") + "\n\n")
        f.write("nDCG@10 uses graded gains E=1.0, S=0.1, C=0.01, I=0 in both tables.\n")
    print(open(RESULTS_DIR / "metrics.md").read())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, setting in zip(axes, SETTINGS):
        for name in runs:
            ax.plot(RECALL_LEVELS, curves[(setting, name)], marker="o", ms=3, label=name)
        rel = " or ".join(sorted(SETTINGS[setting]))
        ax.set(title=f"{setting.capitalize()} ({rel} relevant)", xlabel="Recall",
               xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Interpolated precision")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"11-point interpolated precision-recall (top {K}, full-catalog retrieval)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pr_curve.png", dpi=150)
    pd.DataFrame({f"{s}|{n}": c for (s, n), c in curves.items()},
                 index=RECALL_LEVELS).to_csv(RESULTS_DIR / "pr_curve.csv", float_format="%.4f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=None, help="sample N test queries")
    args = ap.parse_args()

    queries, qrels = load_eval_set("test", max_queries=args.max_queries)
    print(f"evaluating on {len(queries):,} test queries")
    runs, latency = run_rankers(queries)
    (ARTIFACTS_DIR / "runs.json").write_text(json.dumps(runs))
    write_results(runs, latency, qrels)


if __name__ == "__main__":
    main()
