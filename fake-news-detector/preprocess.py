"""
preprocess.py
-------------
Text cleaning utilities for the Fake News Detector project.

Steps performed:
  1. Convert text to lowercase.
  2. Remove URLs, HTML tags, punctuation and digits.
  3. Tokenize and remove English stopwords.
  4. (Optionally) lemmatize / stem tokens.

The cleaned text is later passed to a TF-IDF vectorizer in train.py / app.py.
"""

import re
import string

import nltk
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# Make sure NLTK resources are available the first time the module is imported.
# ---------------------------------------------------------------------------
def _ensure_nltk_resources() -> None:
    for resource in ("stopwords",):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_resources()

# A frozen set is faster for "in" look-ups than a list.
STOPWORDS = frozenset(stopwords.words("english"))

# Pre-compile regexes once -- saves a lot of time on large datasets.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Clean a single text string and return the processed version.

    Parameters
    ----------
    text : str
        Raw input text (headline + article body, for example).

    Returns
    -------
    str
        Lower-cased, stop-word-free, punctuation-free text.
    """
    if text is None:
        return ""

    # 1) Make sure we have a string and normalise case.
    text = str(text).lower()

    # 2) Strip URLs and HTML tags.
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)

    # 3) Remove punctuation.
    text = text.translate(_PUNCT_TABLE)

    # 4) Keep only alphabetic characters and whitespace.
    text = _NON_ALPHA_RE.sub(" ", text)

    # 5) Tokenize on whitespace, drop stopwords + very short tokens.
    tokens = [tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 2]

    # 6) Collapse multiple spaces.
    return _MULTISPACE_RE.sub(" ", " ".join(tokens)).strip()


def clean_series(series):
    """Vectorised helper to clean a pandas Series of strings."""
    return series.fillna("").astype(str).map(clean_text)
