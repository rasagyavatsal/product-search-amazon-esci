# Product Search using Amazon ESCI

An information-retrieval system for e-commerce product search over the
[Amazon ESCI Shopping Queries Dataset](https://github.com/amazon-science/esci-data)
([paper](https://arxiv.org/abs/2206.06588)). It compares lexical (TF-IDF, BM25),
neural (Sentence Transformer) and hybrid (Reciprocal Rank Fusion) ranking.

## Dataset

Each (query, product) pair is labelled **E**xact, **S**ubstitute, **C**omplement
or **I**rrelevant. This project uses the **US locale, small version** (the
Task 1 ranking subset):

| | |
|---|---|
| Catalog | 482,105 products (every product judged for any US small-version query) |
| Test queries | 8,956 |
| Test judgments | 181,701 |
| Label mix (US small) | E 43.5% · S 35.1% · I 16.9% · C 4.5% |

Products are indexed on `title + brand + color`.

## Pipeline

1. **Preprocessing** (`esci/preprocess.py`): HTML unescape and tag removal,
   lowercasing, punctuation removal, whitespace tokenization, stopword removal,
   Snowball (Porter2) stemming.
2. **Indexing** (`esci/build_index.py`):
   - TF-IDF: sublinear-TF, L2-normalised sparse matrix (scikit-learn)
   - BM25: `bm25s` (k1=1.2, b=0.75)
   - Sentence Transformer: `all-MiniLM-L6-v2` embeddings (384-d, float16)
     over lightly cleaned, unstemmed text
3. **Ranking** (`esci/rankers.py`): TF-IDF cosine, BM25, and dense cosine
   similarity (exact search).
4. **Hybrid** (`esci/fusion.py`): Reciprocal Rank Fusion,
   `score(d) = Σ 1 / (60 + rank_i(d))`, over the top 100 of each ranker.
5. **Evaluation** (`esci/evaluate.py`): P@10, Recall@100, MAP@100, nDCG@10,
   11-point interpolated precision-recall curve.
6. **Search interface** (`app.py`): Streamlit app.

## Evaluation setup

- **Full-catalog retrieval:** every method retrieves the top 100 of all 482K
  products for each test query.
- **Binary relevance** (P@10, Recall, MAP, PR curve) is reported two ways:
  *strict* (Exact only) and *lenient* (Exact or Substitute).
- **nDCG@10** uses the official KDD Cup 2022 graded gains:
  E=1.0, S=0.1, C=0.01, I=0.
- ESCI only judges about 20 products per query. Retrieved products without a
  judgment count as non-relevant, so absolute scores are a **lower bound**.
  Comparing methods against each other is still fair.

## Results

RESULTS_PLACEHOLDER

## Usage

```bash
uv sync

# 1. download data (~1.3 GB) into data/
mkdir -p data && cd data
for f in shopping_queries_dataset_examples.parquet shopping_queries_dataset_products.parquet; do
  curl -L -O "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/$f"
done
cd ..

# 2. build indexes (catalog, TF-IDF, BM25, embeddings)
uv run python -m esci.build_index

# 3. evaluate -> results/metrics.md, results/pr_curve.png
uv run python -m esci.evaluate          # add --max-queries 1000 for a quick run

# 4. search interface
uv run streamlit run app.py

# tests
uv run pytest
```

## Layout

```
esci/
  preprocess.py   cleaning, tokenization, stemming
  data.py         dataset loading, catalog construction, qrels
  rankers.py      TF-IDF, BM25, Sentence Transformer
  fusion.py       Reciprocal Rank Fusion
  metrics.py      P@k, Recall@k, AP, nDCG, interpolated PR
  build_index.py  builds artifacts/
  evaluate.py     writes results/
app.py            Streamlit UI
tests/            unit tests
results/          metrics tables + PR curve
```
