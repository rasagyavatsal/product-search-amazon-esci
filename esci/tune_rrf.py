"""Tune weighted RRF (BM25 + Sentence Transformer) on held-out train queries.

The test split is never touched here: weights are chosen on a sample of train
queries (no model in this project is trained on them), then esci.evaluate
reports the chosen setting on the test split.

Usage: uv run python -m esci.tune_rrf [--dev-queries 2000]
"""

import argparse
import itertools
import json

import pandas as pd

from esci.data import RESULTS_DIR, load_eval_set
from esci.evaluate import K, RRF_BEST_PATH, run_base_rankers
from esci.fusion import rrf
from esci.metrics import evaluate_run

BM25_WEIGHTS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
RRF_KS = [5, 10, 20, 40, 60, 100]
SELECT_BY = "nDCG@10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-queries", type=int, default=2000)
    args = ap.parse_args()

    queries, qrels = load_eval_set("train", max_queries=args.dev_queries, seed=42)
    print(f"tuning on {len(queries):,} train queries")
    runs, _ = run_base_rankers(queries, names={"BM25", "SentenceTransformer"})
    bm25, dense = runs["BM25"], runs["SentenceTransformer"]

    rows = []
    for w, k in itertools.product(BM25_WEIGHTS, RRF_KS):
        fused = {q: rrf([bm25[q], dense[q]], k=k, weights=[w, 1.0], top_n=K) for q in queries}
        m = evaluate_run(fused, qrels, r_k=K)
        rows.append({"bm25_weight": w, "k": k, **{n: m[n] for n in
                     ["P@10", "Recall@100", "MAP", "nDCG@10"]}})
    for name, run in [("BM25", bm25), ("SentenceTransformer", dense)]:
        m = evaluate_run(run, qrels, r_k=K)
        print(f"{name:>20}: " + "  ".join(f"{n} {m[n]:.4f}" for n in
                                           ["P@10", "Recall@100", "MAP", "nDCG@10"]))

    df = pd.DataFrame(rows).sort_values(SELECT_BY, ascending=False)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "rrf_tuning.csv", index=False, float_format="%.4f")
    print(df.head(10).to_string(index=False, float_format="%.4f"))

    best = df.iloc[0]
    config = {"weights": [float(best.bm25_weight), 1.0], "k": int(best.k),
              "selected_by": SELECT_BY, "dev_queries": len(queries), "dev_split": "train"}
    RRF_BEST_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"best: {config}")


if __name__ == "__main__":
    main()
