import string

from pl2016.text import DIGITS, PUNCTUATION, SYMBOLS, clean_text, get_tokens
from pl2016 import text as text_module

# Notebook Step 3, the first sentence of NY CPL § 450.30, and the token list
# printed under it. Digits in "450.10" are stripped on purpose.
ORACLE_SENTENCE = (
    "An  appeal  by  the  defendant from a sentence, as authorized by  "
    "subdivision two of section 450.10, may be based  upon  the  ground  "
    "that  such sentence either was (a) invalid as a matter of law, or (b) "
    "harsh or excessive."
)
NOTEBOOK_TOKENS = [
    "appeal",
    "defend",
    "sentenc",
    "author",
    "subdivis",
    "two",
    "section",
    "may",
    "base",
    "upon",
    "ground",
    "sentenc",
    "either",
    "invalid",
    "matter",
    "law",
    "harsh",
    "excess",
]


def test_punctuation_and_digits_match_the_notebook():
    assert PUNCTUATION == string.punctuation
    assert DIGITS == "0123456789"
    assert SYMBOLS[0] == "\n"
    assert SYMBOLS[2] == "\\u"


def test_digits_are_stripped():
    assert "40" not in clean_text("overtime after 40 hours")
    assert "40" not in get_tokens("I worked 40 hours of overtime")


def test_get_tokens_returns_one_string_and_a_second_join_breaks_it():
    tokens = get_tokens("Unpaid overtime after the shift")
    assert isinstance(tokens, str)
    assert tokens.split()
    assert " ".join(tokens) != tokens


def test_one_stemmer_is_reused():
    get_tokens("appeal from a sentence")
    stemmer = text_module._STEMMER
    stops = text_module._STOPWORDS
    get_tokens("another sentence")
    assert text_module._STEMMER is stemmer
    assert text_module._STOPWORDS is stops


def test_notebook_sentence_tokens():
    # The printed list drops the leftover "b" from "(b)". "(a)" vanishes because
    # "a" is a stopword. NLTK 3.10.3 matches every other printed token, including
    # "upon". Recorded in REBUILD_NOTES. Do not drop single letters to match it.
    actual = get_tokens(ORACLE_SENTENCE).split()
    assert actual == NOTEBOOK_TOKENS[:16] + ["b"] + NOTEBOOK_TOKENS[16:]
