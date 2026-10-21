from esci.preprocess import analyze
from esci.rankers import BM25Ranker, TfidfRanker

IDS = ["p1", "p2", "p3", "p4"]
DOCS = ["Red running shoes for men", "Blue iPhone 13 case",
        "Red leather iPhone case", "Stainless steel water bottle"]


def _ids(hits):
    return [pid for pid, _ in hits]


def test_tfidf_ranks_matching_docs(tmp_path):
    ranker = TfidfRanker.build(IDS, [analyze(d) for d in DOCS])
    ranker.save(tmp_path)
    hits = TfidfRanker.load(tmp_path).search(["iphone case red", "zzz"], k=3)
    assert _ids(hits[0])[0] == "p3" and _ids(hits[0])[1] == "p2"  # p1 matches only "red"
    assert hits[1] == []


def test_bm25_ranks_matching_docs(tmp_path):
    ranker = BM25Ranker.build(IDS, [analyze(d) for d in DOCS])
    ranker.save(tmp_path)
    hits = BM25Ranker.load(tmp_path).search(["iphone case red", "water bottles"], k=3)
    assert _ids(hits[0])[0] == "p3"
    assert _ids(hits[1]) == ["p4"]
