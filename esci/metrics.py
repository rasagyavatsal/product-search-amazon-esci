"""IR evaluation metrics over ESCI relevance judgments.

Binary metrics (P@k, Recall@k, AP) treat a product as relevant when its ESCI
label is in ``rel_labels`` (default: Exact only). nDCG uses graded gains.
Products a system retrieves that have no judgment count as non-relevant.
"""

import numpy as np

# Official KDD Cup 2022 gains for ESCI.
ESCI_GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
RECALL_LEVELS = np.linspace(0.0, 1.0, 11)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def average_precision(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def ndcg_at_k(ranked: list[str], labels: dict[str, str], k: int,
              gains: dict[str, float] = ESCI_GAINS) -> float:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    got = np.array([gains.get(labels.get(d, "I"), 0.0) for d in ranked[:k]])
    dcg = float((got * discounts[: len(got)]).sum())
    ideal = np.sort([gains[l] for l in labels.values()])[::-1][:k]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def interpolated_pr(ranked: list[str], relevant: set[str]) -> np.ndarray:
    """11-point interpolated precision at recall 0.0, 0.1, ..., 1.0."""
    precisions, recalls, hits = [], [], 0
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
        precisions.append(hits / i)
        recalls.append(hits / len(relevant))
    precisions, recalls = np.array(precisions), np.array(recalls)
    out = np.zeros(len(RECALL_LEVELS))
    for j, r in enumerate(RECALL_LEVELS):
        mask = recalls >= r
        out[j] = precisions[mask].max() if mask.any() else 0.0
    return out


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, str]],
                 rel_labels: frozenset[str] = frozenset("E"),
                 p_k: int = 10, r_k: int = 100, ndcg_k: int = 10) -> dict:
    """Average metrics over queries that have at least one relevant product.

    ``run`` maps query_id -> ranked product ids; ``qrels`` maps
    query_id -> {product_id: esci_label}.
    """
    p, r, ap, nd, pr = [], [], [], [], []
    for qid, labels in qrels.items():
        relevant = {d for d, l in labels.items() if l in rel_labels}
        if not relevant:
            continue
        ranked = run.get(qid, [])
        p.append(precision_at_k(ranked, relevant, p_k))
        r.append(recall_at_k(ranked, relevant, r_k))
        ap.append(average_precision(ranked, relevant, r_k))
        nd.append(ndcg_at_k(ranked, labels, ndcg_k))
        pr.append(interpolated_pr(ranked[:r_k], relevant))
    return {
        f"P@{p_k}": float(np.mean(p)),
        f"Recall@{r_k}": float(np.mean(r)),
        "MAP": float(np.mean(ap)),
        f"nDCG@{ndcg_k}": float(np.mean(nd)),
        "pr_curve": np.mean(pr, axis=0),
        "n_queries": len(p),
    }
