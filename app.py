"""Streamlit search interface over the ESCI US product catalog.

Usage: uv run streamlit run app.py
"""

import pandas as pd
import streamlit as st

from esci.data import ARTIFACTS_DIR, RESULTS_DIR, load_catalog, load_eval_set
from esci.fusion import rrf
from esci.metrics import ndcg_at_k, precision_at_k
from esci.rankers import BM25Ranker, DenseRanker, TfidfRanker

LABEL_NAMES = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}
LABEL_COLORS = {"E": "#1a7f37", "S": "#9a6700", "C": "#0969da", "I": "#cf222e"}
METHODS = ["RRF (BM25 + ST)", "RRF (TF-IDF + BM25 + ST)", "BM25", "SentenceTransformer", "TF-IDF"]

st.set_page_config(page_title="ESCI Product Search", layout="wide")


@st.cache_resource(show_spinner="Loading indexes...")
def load_rankers():
    rankers = {}
    for cls, sub in [(TfidfRanker, "tfidf"), (BM25Ranker, "bm25"), (DenseRanker, "dense")]:
        if (ARTIFACTS_DIR / sub).exists():
            rankers[cls.name] = cls.load(ARTIFACTS_DIR / sub)
    return rankers


@st.cache_data(show_spinner="Loading catalog...")
def load_products():
    return load_catalog().set_index("product_id")


@st.cache_data(show_spinner=False)
def load_test_queries():
    queries, qrels = load_eval_set("test")
    by_text = {}
    for qid, text in queries.items():
        by_text.setdefault(text, qrels[qid])
    return by_text


def search(rankers, method: str, query: str, k: int) -> list[tuple[str, float]]:
    if method.startswith("RRF"):
        parts = ["BM25", "SentenceTransformer"] + (["TF-IDF"] if "TF-IDF" in method else [])
        lists = [[pid for pid, _ in rankers[p].search([query], k=100)[0]] for p in parts]
        fused = rrf(lists, k=60, top_n=k)
        return [(pid, None) for pid in fused]
    return rankers[method].search([query], k=k)[0]


rankers = load_rankers()
if not rankers:
    st.error("No indexes found. Run `uv run python -m esci.build_index` first.")
    st.stop()
products = load_products()
test_queries = load_test_queries()
available = [m for m in METHODS
             if m in rankers or (m.startswith("RRF") and {"BM25", "SentenceTransformer"} <= rankers.keys())]

st.title("Amazon ESCI Product Search")
st.caption(f"{len(products):,} products (US locale) · TF-IDF, BM25, Sentence Transformer and RRF hybrid")

with st.sidebar:
    method = st.radio("Ranking method", available)
    k = st.slider("Results", 5, 50, 10)
    st.divider()
    example = st.selectbox("Or pick a labelled test query", [""] + sorted(test_queries),
                           help="Test queries show the ESCI judgment for each result.")

query = st.text_input("Search products", value=example, placeholder="e.g. iphone 13 case")

tab_search, tab_results = st.tabs(["Search", "Evaluation results"])

with tab_search:
    if query.strip():
        hits = search(rankers, method, query, k)
        labels = test_queries.get(query, {})
        if labels:
            ranked = [pid for pid, _ in hits]
            exact = {p for p, l in labels.items() if l == "E"}
            c1, c2, c3 = st.columns(3)
            c1.metric("P@10 (Exact)", f"{precision_at_k(ranked, exact, 10):.2f}")
            c2.metric("nDCG@10", f"{ndcg_at_k(ranked, labels, 10):.3f}")
            c3.metric("Judged products", len(labels))
        if not hits:
            st.info("No matching products.")
        for rank, (pid, score) in enumerate(hits, start=1):
            p = products.loc[pid]
            label = labels.get(pid)
            badge = (f"<span style='background:{LABEL_COLORS[label]};color:white;padding:1px 8px;"
                     f"border-radius:10px;font-size:0.8em'>{LABEL_NAMES[label]}</span>"
                     if label else ("<span style='color:gray;font-size:0.8em'>unjudged</span>"
                                    if labels else ""))
            meta = " · ".join(x for x in [p.product_brand, p.product_color] if x)
            score_txt = f" · score {score:.3f}" if score is not None else ""
            st.markdown(
                f"**{rank}. [{p.product_title}](https://www.amazon.com/dp/{pid})** {badge}  \n"
                f"<span style='color:gray;font-size:0.85em'>{meta} · {pid}{score_txt}</span>",
                unsafe_allow_html=True)

with tab_results:
    md, png = RESULTS_DIR / "metrics.md", RESULTS_DIR / "pr_curve.png"
    if md.exists():
        st.markdown(md.read_text())
    if png.exists():
        st.image(str(png))
    if not md.exists():
        st.info("Run `uv run python -m esci.evaluate` to produce the comparison table.")
