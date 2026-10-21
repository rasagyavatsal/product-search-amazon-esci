"""Loading the ESCI dataset: examples (judgments), product catalog, queries."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

EXAMPLES_PATH = DATA_DIR / "shopping_queries_dataset_examples.parquet"
PRODUCTS_PATH = DATA_DIR / "shopping_queries_dataset_products.parquet"
CATALOG_PATH = ARTIFACTS_DIR / "catalog.parquet"

LOCALE = "us"
PRODUCT_COLUMNS = ["product_id", "product_title", "product_brand", "product_color"]


def load_examples(locale: str = LOCALE) -> pd.DataFrame:
    """Judgments for the small (Task 1 / ranking) version of the dataset."""
    df = pd.read_parquet(EXAMPLES_PATH, filters=[("product_locale", "==", locale),
                                                 ("small_version", "==", 1)])
    return df[["query_id", "query", "product_id", "esci_label", "split"]]


def build_catalog(locale: str = LOCALE) -> pd.DataFrame:
    """Every product judged for any small-version query, with searchable text."""
    ids = set(load_examples(locale).product_id)
    table = pq.read_table(PRODUCTS_PATH, columns=PRODUCT_COLUMNS + ["product_locale"],
                          filters=[("product_locale", "==", locale)])
    products = table.drop(["product_locale"]).to_pandas()
    products = products[products.product_id.isin(ids)].drop_duplicates("product_id")
    products = products.fillna("").reset_index(drop=True)
    products["text"] = (products.product_title + " " + products.product_brand + " "
                        + products.product_color).str.strip()
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    products.to_parquet(CATALOG_PATH, index=False)
    return products


def load_catalog() -> pd.DataFrame:
    return pd.read_parquet(CATALOG_PATH)


def load_eval_set(split: str = "test", max_queries: int | None = None,
                  seed: int = 0) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Return (queries: qid -> text, qrels: qid -> {product_id: label})."""
    df = load_examples()
    df = df[df.split == split]
    qids = df.query_id.drop_duplicates()
    if max_queries and max_queries < len(qids):
        qids = qids.sample(max_queries, random_state=seed)
        df = df[df.query_id.isin(set(qids))]
    queries = {str(q): t for q, t in df.groupby("query_id")["query"].first().items()}
    qrels: dict[str, dict[str, str]] = {}
    for q, pid, label in zip(df.query_id, df.product_id, df.esci_label):
        qrels.setdefault(str(q), {})[pid] = label
    return queries, qrels
