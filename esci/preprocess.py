"""Text preprocessing: cleaning, tokenization, stopword removal and stemming."""

import html
import re

import Stemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_STEMMER = Stemmer.Stemmer("english")  # Snowball (Porter2)
STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


def clean(text: str) -> str:
    """Lowercase, unescape HTML entities, strip tags and punctuation."""
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text.lower())
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Clean, split on whitespace and drop stopwords."""
    return [t for t in clean(text).split() if t not in STOPWORDS]


def stem(tokens: list[str]) -> list[str]:
    return _STEMMER.stemWords(tokens)


def analyze(text: str) -> list[str]:
    """Full lexical pipeline used by TF-IDF and BM25: clean -> tokenize -> stem."""
    return stem(tokenize(text))
