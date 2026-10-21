"""Paired significance test between two methods' runs on the test split.

Usage: uv run python -m esci.significance ["Weighted RRF (BM25 + ST)"] ["BM25"]
Requires artifacts/runs.json from esci.evaluate.
"""

import json
import sys

import numpy as np
from scipy import stats

from esci.data import ARTIFACTS_DIR, load_eval_set
from esci.metrics import average_precision, ndcg_at_k


def main():
    a_name = sys.argv[1] if len(sys.argv) > 1 else "Weighted RRF (BM25 + ST)"
    b_name = sys.argv[2] if len(sys.argv) > 2 else "BM25"
    runs = json.loads((ARTIFACTS_DIR / "runs.json").read_text())
    _, qrels = load_eval_set("test")
    qrels = {q: l for q, l in qrels.items() if "E" in l.values()}
    metrics = {
        "nDCG@10": lambda run, q, l: ndcg_at_k(run[q], l, 10),
        "MAP": lambda run, q, l: average_precision(run[q], {d for d, x in l.items() if x == "E"}, 100),
    }
    print(f"{a_name} vs {b_name}, {len(qrels):,} queries")
    rng = np.random.default_rng(0)
    for name, f in metrics.items():
        a = np.array([f(runs[a_name], q, l) for q, l in qrels.items()])
        b = np.array([f(runs[b_name], q, l) for q, l in qrels.items()])
        d = a - b
        boots = rng.choice(d, (2000, len(d))).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {name}: {d.mean():+.4f}  95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]  "
              f"paired t-test p={stats.ttest_rel(a, b).pvalue:.1e}  "
              f"wins/losses/ties {(d > 0).sum()}/{(d < 0).sum()}/{(d == 0).sum()}")


if __name__ == "__main__":
    main()
