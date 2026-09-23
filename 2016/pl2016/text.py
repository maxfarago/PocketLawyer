"""clean_text and get_tokens. Digit stripping stays; it is a 2016 decision.

The notebook built a new Porter stemmer and a new stopword list on every token.
Both are created once, on the first call.
"""

from __future__ import annotations

import string

# Python 2 kept '\u' and '\u201' as literal backslash text. Python 3 rejects
# those as unicode escapes, so they are written with escaped backslashes.
# '\u' is applied first, which is why '\u201' never matches on its own.
SYMBOLS = ("\n", "\r", "\\u", "\\u201", "Nolo.com", "nolo.com")
PUNCTUATION = string.punctuation
DIGITS = "0123456789"

_STEMMER = None
_STOPWORDS: frozenset[str] | None = None


def clean_text(text: str) -> str:
    for symbol in SYMBOLS:
        text = text.replace(symbol, " ")
    for mark in PUNCTUATION:
        text = text.replace(mark, "")
    for digit in DIGITS:
        text = text.replace(digit, "")
    return text.lower()


def get_tokens(text: str) -> str:
    """Return one space-joined string of stems. Do not join this result again."""
    from nltk.tokenize import word_tokenize

    stemmer, stops = _resources()
    words = word_tokenize(clean_text(text))
    return " ".join(stemmer.stem(word) for word in words if word not in stops)


def _resources():
    global _STEMMER, _STOPWORDS
    if _STEMMER is not None and _STOPWORDS is not None:
        return _STEMMER, _STOPWORDS
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    for resource, path in (("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
    _STEMMER = PorterStemmer()
    _STOPWORDS = frozenset(stopwords.words("english"))
    return _STEMMER, _STOPWORDS
