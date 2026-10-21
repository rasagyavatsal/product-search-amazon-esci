"""Ranking models over the product catalog: TF-IDF, BM25 and a sentence transformer.

Each ranker is built from the catalog, saved under ``artifacts/`` and exposes
``search(queries, k) -> list[list[tuple[product_id, score]]]``.
"""

import json
from pathlib import Path

import bm25s
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from esci.preprocess import analyze, clean

Hits = list[list[tuple[str, float]]]
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _identity(tokens):
    return tokens


def _top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the k largest entries in each row, sorted descending."""
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)


class TfidfRanker:
    name = "TF-IDF"

    def __init__(self, product_ids: list[str], vectorizer: TfidfVectorizer, matrix: sp.csr_matrix):
        self.product_ids = np.asarray(product_ids)
        self.vectorizer = vectorizer
        self.matrix_t = matrix.T.tocsr()  # (vocab, docs) for fast csr @ csr products

    @classmethod
    def build(cls, product_ids, doc_tokens):
        vec = TfidfVectorizer(analyzer=_identity, sublinear_tf=True, dtype=np.float32)
        return cls(product_ids, vec, vec.fit_transform(doc_tokens).tocsr())

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path / "vectorizer.joblib")
        sp.save_npz(path / "matrix.npz", self.matrix_t.T.tocsr())
        np.save(path / "product_ids.npy", self.product_ids)

    @classmethod
    def load(cls, path: Path):
        return cls(np.load(path / "product_ids.npy", allow_pickle=True),
                   joblib.load(path / "vectorizer.joblib"), sp.load_npz(path / "matrix.npz"))

    def search(self, queries: list[str], k: int = 100, batch_size: int = 256) -> Hits:
        out: Hits = []
        for i in range(0, len(queries), batch_size):
            q = self.vectorizer.transform([analyze(t) for t in queries[i:i + batch_size]])
            scores = (q @ self.matrix_t).tocsr()  # cosine: rows are L2-normalised
            for row in range(scores.shape[0]):
                start, end = scores.indptr[row], scores.indptr[row + 1]
                data, cols = scores.data[start:end], scores.indices[start:end]
                top = np.argsort(-data)[:k]
                out.append([(str(self.product_ids[cols[j]]), float(data[j])) for j in top])
        return out


class BM25Ranker:
    name = "BM25"

    def __init__(self, product_ids: list[str], retriever: bm25s.BM25):
        self.product_ids = np.asarray(product_ids)
        self.retriever = retriever

    @classmethod
    def build(cls, product_ids, doc_tokens, k1: float = 1.2, b: float = 0.75):
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(doc_tokens, show_progress=False)
        return cls(product_ids, retriever)

    def save(self, path: Path):
        self.retriever.save(path, show_progress=False)
        np.save(path / "product_ids.npy", self.product_ids)

    @classmethod
    def load(cls, path: Path):
        return cls(np.load(path / "product_ids.npy", allow_pickle=True),
                   bm25s.BM25.load(path, show_progress=False))

    def search(self, queries: list[str], k: int = 100) -> Hits:
        docs, scores = self.retriever.retrieve([analyze(t) for t in queries], k=k,
                                               show_progress=False, n_threads=-1)
        # bm25s pads with arbitrary zero-score documents when few terms match.
        return [[(str(self.product_ids[d]), float(s)) for d, s in zip(drow, srow) if s > 0]
                for drow, srow in zip(docs, scores)]


class DenseRanker:
    name = "SentenceTransformer"

    def __init__(self, product_ids: list[str], embeddings: np.ndarray, model_name: str = DENSE_MODEL):
        self.product_ids = np.asarray(product_ids)
        self.embeddings = embeddings  # (docs, dim), L2-normalised, float16
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @classmethod
    def build(cls, product_ids, texts: list[str], model_name: str = DENSE_MODEL,
              batch_size: int = 256):
        ranker = cls(product_ids, np.empty((0, 0), dtype=np.float16), model_name)
        emb = ranker.model.encode([clean(t) for t in texts], batch_size=batch_size,
                                  normalize_embeddings=True, show_progress_bar=True,
                                  convert_to_numpy=True)
        ranker.embeddings = emb.astype(np.float16)
        return ranker

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        np.save(path / "product_ids.npy", self.product_ids)
        (path / "model.json").write_text(json.dumps({"model_name": self.model_name}))

    @classmethod
    def load(cls, path: Path):
        meta = json.loads((path / "model.json").read_text())
        return cls(np.load(path / "product_ids.npy", allow_pickle=True),
                   np.load(path / "embeddings.npy", mmap_mode="r"), meta["model_name"])

    def search(self, queries: list[str], k: int = 100, batch_size: int = 64,
               doc_chunk: int = 100_000) -> Hits:
        q = self.model.encode([clean(t) for t in queries], batch_size=256,
                              normalize_embeddings=True, convert_to_numpy=True)
        out: Hits = []
        for i in range(0, len(q), batch_size):
            qb = q[i:i + batch_size].astype(np.float32)
            # Top-k per document chunk, then merge, to bound memory on large catalogs.
            cand_idx, cand_val = [], []
            for j in range(0, len(self.embeddings), doc_chunk):
                scores = qb @ np.asarray(self.embeddings[j:j + doc_chunk], dtype=np.float32).T
                idx, val = _top_k(scores, k)
                cand_idx.append(idx + j)
                cand_val.append(val)
            idx, val = np.hstack(cand_idx), np.hstack(cand_val)
            top, vals = _top_k(val, k)
            idx = np.take_along_axis(idx, top, axis=1)
            out.extend([[(str(self.product_ids[d]), float(s)) for d, s in zip(drow, srow)]
                        for drow, srow in zip(idx, vals)])
        return out
