"""Build the product catalog and all search indexes.

Usage: uv run python -m esci.build_index [--skip-dense] [--rebuild-catalog]
"""

import argparse
import gc
import time

from esci.data import ARTIFACTS_DIR, CATALOG_PATH, build_catalog, load_catalog
from esci.preprocess import analyze
from esci.rankers import BM25Ranker, DenseRanker, TfidfRanker


def _timed(label):
    class _T:
        def __enter__(self):
            self.t = time.perf_counter()
            print(f"[{label}] ...", flush=True)

        def __exit__(self, *exc):
            print(f"[{label}] done in {time.perf_counter() - self.t:.1f}s", flush=True)
    return _T()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dense", action="store_true", help="skip sentence-transformer encoding")
    ap.add_argument("--dense-only", action="store_true", help="only build the dense index")
    ap.add_argument("--rebuild-catalog", action="store_true")
    args = ap.parse_args()

    with _timed("catalog"):
        catalog = build_catalog() if args.rebuild_catalog or not CATALOG_PATH.exists() else load_catalog()
    print(f"catalog: {len(catalog):,} products")
    ids = catalog.product_id.tolist()

    if not args.dense_only:
        with _timed("tokenize + stem"):
            tokens = [analyze(t) for t in catalog.text]
        with _timed("TF-IDF index"):
            TfidfRanker.build(ids, tokens).save(ARTIFACTS_DIR / "tfidf")
        gc.collect()
        with _timed("BM25 index"):
            BM25Ranker.build(ids, tokens).save(ARTIFACTS_DIR / "bm25")
        del tokens
        gc.collect()

    if not args.skip_dense:
        with _timed("dense embeddings"):
            DenseRanker.build(ids, catalog.text.tolist()).save(ARTIFACTS_DIR / "dense")


if __name__ == "__main__":
    main()
