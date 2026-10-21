import numpy as np
import pytest

from esci.fusion import rrf
from esci.metrics import (average_precision, evaluate_run, interpolated_pr,
                          ndcg_at_k, precision_at_k, recall_at_k)

RANKED = ["a", "b", "c", "d"]
RELEVANT = {"a", "c", "z"}


def test_precision_and_recall():
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(2 / 3)


def test_average_precision():
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / 3 relevant
    assert average_precision(RANKED, RELEVANT, 4) == pytest.approx((1 + 2 / 3) / 3)


def test_ndcg_perfect_and_graded():
    labels = {"a": "E", "b": "S", "c": "I"}
    assert ndcg_at_k(["a", "b", "c"], labels, 3) == pytest.approx(1.0)
    swapped = ndcg_at_k(["b", "a", "c"], labels, 3)
    expected = (0.1 + 1 / np.log2(3)) / (1 + 0.1 / np.log2(3))
    assert swapped == pytest.approx(expected)


def test_ndcg_unjudged_counts_as_zero():
    assert ndcg_at_k(["x", "a"], {"a": "E"}, 2) == pytest.approx(1 / np.log2(3))


def test_interpolated_pr_is_monotone():
    curve = interpolated_pr(RANKED, RELEVANT)
    assert curve[0] == 1.0 and curve[-1] == 0.0  # "z" never retrieved
    assert all(np.diff(curve) <= 0)


def test_evaluate_run_skips_queries_without_relevant():
    qrels = {"q1": {"a": "E", "b": "I"}, "q2": {"c": "S"}}
    res = evaluate_run({"q1": ["a", "b"], "q2": ["c"]}, qrels)
    assert res["n_queries"] == 1 and res["MAP"] == 1.0


def test_rrf():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]], k=60)
    assert fused[0] == "a"  # 1/61 + 1/62 beats c's 1/63 + 1/61
    assert set(fused) == {"a", "b", "c", "d"}
    assert rrf([["a", "b"]], top_n=1) == ["a"]
