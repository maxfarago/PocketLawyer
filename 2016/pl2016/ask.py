"""Query path: one predicted flair, then top-5 statutes and top-5 articles.

Serving reads joblib artifacts. It does not query Mongo. Section scores are
classifier scores from MultinomialNB, not calibrated probabilities. Law and
article scores are cosine similarity. Higher is closer.

    python -c "from pl2016.ask import ask; print(ask('overtime pay', 'NY'))"
"""

from __future__ import annotations

import time
from pathlib import Path

from pl2016 import __version__
from pl2016.config import STATES
from pl2016.posts.reddit import LABELS
from pl2016.text import get_tokens
from pl2016.train import ARTIFACTS, slug

TOP_LAWS = 5
TOP_ARTICLES = 5
TOP_SCORES = 3
SNIPPET_CHARS = 400

# Notebook state list, without "National", which is not a jurisdiction and would
# drop ordinary titles. Longest name first so "West Virginia" is not "Virginia"
# and "Arkansas" is not "Kansas".
_STATE_NAMES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "PR": "Puerto Rico",
    "GU": "Guam",
    "VI": "Virgin Islands",
}
STATE_NAMES = tuple(sorted(_STATE_NAMES.items(), key=lambda item: len(item[1]), reverse=True))
_ACCEPTED = {"ny": "NY", "new york": "NY", "ca": "CA", "california": "CA"}


class UnsupportedState(ValueError):
    """The caller named a state this version does not rank."""


def normalize_state(state: str) -> str:
    code = _ACCEPTED.get(state.strip().casefold())
    if code is None:
        raise UnsupportedState("This version knows New York and California.")
    return code


def states_in_title(title: str) -> list[str]:
    """Jurisdiction codes named in a title. Longer names consume shorter ones."""
    remaining = title.casefold()
    found: list[str] = []
    for code, name in STATE_NAMES:
        token = name.casefold()
        if token in remaining:
            found.append(code)
            remaining = remaining.replace(token, " ")
    return found


def snippet(text: str) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= SNIPPET_CHARS:
        return collapsed
    cut = collapsed[:SNIPPET_CHARS]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut


def select_articles(ranked: list[tuple[float, dict]], user_state: str) -> list[dict]:
    """User's state first, then titles that name no state. Other states drop out."""
    preferred: list[dict] = []
    neutral: list[dict] = []
    for score, row in ranked:
        named = states_in_title(row.get("title") or "")
        item = {
            "title": row.get("title") or "",
            "url": row.get("url") or "",
            "score": round(float(score), 4),
        }
        if not named:
            neutral.append(item)
        elif any(code != user_state for code in named):
            continue
        else:
            preferred.append(item)
    return (preferred + neutral)[:TOP_ARTICLES]


def _cosine_order(vectorizer, matrix, tokens: str) -> list[tuple[float, int]]:
    from sklearn.metrics.pairwise import cosine_similarity

    user = vectorizer.transform([tokens])
    scores = cosine_similarity(matrix, user).ravel()
    order = scores.argsort()[::-1]
    return [(float(scores[index]), int(index)) for index in order]


class Models:
    """Loaded artifacts. One instance serves every request."""

    def __init__(self, classifier, laws: dict, articles: dict):
        self.classifier = classifier
        self.laws = laws
        self.articles = articles

    @classmethod
    def load(cls, directory: Path = ARTIFACTS) -> Models:
        import joblib

        classifier = joblib.load(directory / "posts_classifier.joblib")
        laws = {}
        for state in STATES:
            for label in LABELS:
                path = directory / "laws" / state / f"{slug(label)}.joblib"
                laws[(state, label)] = joblib.load(path)
        articles = {}
        for label in LABELS:
            path = directory / "articles" / f"{slug(label)}.joblib"
            if path.exists():
                articles[label] = joblib.load(path)
        return cls(classifier, laws, articles)


def ask(question: str, state: str, models: Models | None = None) -> dict:
    if not (question or "").strip():
        raise ValueError("question is required")
    code = normalize_state(state)
    loaded = models if models is not None else Models.load()
    started = time.perf_counter()
    tokens = get_tokens(question)

    classify_started = time.perf_counter()
    vectorizer = loaded.classifier["vectorizer"]
    classifier = loaded.classifier["classifier"]
    user = vectorizer.transform([tokens])
    probabilities = classifier.predict_proba(user)[0]
    classes = [str(label) for label in classifier.classes_]
    order = sorted(range(len(classes)), key=lambda index: probabilities[index], reverse=True)
    section = classes[order[0]]
    section_probs = [
        {"section": classes[index], "score": round(float(probabilities[index]), 4)}
        for index in order[:TOP_SCORES]
    ]
    classify_ms = _ms(classify_started)

    laws_started = time.perf_counter()
    law_pack = loaded.laws[(code, section)]
    law_order = _cosine_order(law_pack["vectorizer"], law_pack["matrix"], tokens)
    laws = []
    for score, index in law_order[:TOP_LAWS]:
        row = law_pack["rows"][index]
        laws.append(
            {
                "citation": row.get("citation") or "",
                "title": row.get("title") or "",
                "url": row.get("url") or "",
                "snippet": snippet(row.get("text") or ""),
                "score": round(score, 4),
            }
        )
    laws_ms = _ms(laws_started)

    articles_started = time.perf_counter()
    article_pack = loaded.articles.get(section)
    articles: list[dict] = []
    if article_pack is not None:
        article_order = _cosine_order(article_pack["vectorizer"], article_pack["matrix"], tokens)
        ranked = [(score, article_pack["rows"][index]) for score, index in article_order]
        articles = select_articles(ranked, code)
    articles_ms = _ms(articles_started)

    return {
        "version": __version__,
        "state": code,
        "section": section,
        "section_probs": section_probs,
        "laws": laws,
        "articles": articles,
        "timings_ms": {
            "classify": classify_ms,
            "laws": laws_ms,
            "articles": articles_ms,
            "total": _ms(started),
        },
    }


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
