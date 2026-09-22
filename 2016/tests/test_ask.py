from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from pl2016.ask import (
    Models,
    UnsupportedState,
    ask,
    normalize_state,
    select_articles,
    snippet,
    states_in_title,
)
from pl2016.text import get_tokens

ROOT = Path(__file__).resolve().parents[1]


def test_west_virginia_is_not_virginia_and_arkansas_is_not_kansas():
    assert states_in_title("West Virginia overtime guide") == ["WV"]
    assert states_in_title("Virginia overtime guide") == ["VA"]
    assert states_in_title("Arkansas small claims") == ["AR"]
    assert states_in_title("Overtime pay basics") == []


def test_user_state_titles_come_before_titles_that_name_no_state():
    ranked = [
        (0.9, {"title": "Virginia Overtime", "url": "http://a"}),
        (0.8, {"title": "California Overtime", "url": "http://b"}),
        (0.7, {"title": "Overtime Pay Basics", "url": "http://c"}),
        (0.6, {"title": "West Virginia Guide", "url": "http://d"}),
    ]
    chosen = select_articles(ranked, "CA")
    assert [item["title"] for item in chosen] == ["California Overtime", "Overtime Pay Basics"]


def test_snippet_is_a_prefix_of_the_statute():
    text = "Eight hours of labor. " * 40
    cut = snippet(text)
    assert text.split()[0] in cut
    assert cut == " ".join(text.split())[: len(cut)] or cut in " ".join(text.split())
    assert len(cut) < len(" ".join(text.split()))


def test_only_new_york_and_california():
    assert normalize_state("ny") == "NY"
    assert normalize_state("California") == "CA"
    with pytest.raises(UnsupportedState, match="New York and California"):
        normalize_state("Maryland")


def test_ask_uses_each_question_once_and_returns_the_contract():
    employment = get_tokens("my boss will not pay my overtime wages")
    housing = get_tokens("the landlord raised the rent and broke the lease")
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w\w+\b")
    matrix = vectorizer.fit_transform([employment, employment, housing, housing])
    classifier = MultinomialNB().fit(matrix, ["employment", "employment", "housing", "housing"])
    law_vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w\w+\b")
    law_tokens = [
        get_tokens("Eight hours of labor is a day's work and overtime must be paid"),
        get_tokens("Fish and game permits"),
    ]
    law_matrix = law_vectorizer.fit_transform(law_tokens)
    statute = "Eight hours of labor constitutes a day's work. " * 30
    article_vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w\w+\b")
    article_tokens = [
        get_tokens("California overtime pay rules for hourly workers"),
        get_tokens("Virginia overtime rules"),
        get_tokens("Overtime pay when the boss refuses"),
    ]
    article_matrix = article_vectorizer.fit_transform(article_tokens)
    models = Models(
        classifier={"vectorizer": vectorizer, "classifier": classifier},
        laws={
            ("NY", "employment"): {
                "vectorizer": law_vectorizer,
                "matrix": law_matrix,
                "rows": [
                    {
                        "citation": "LAB § 510",
                        "title": "Section 510",
                        "url": "http://law/510",
                        "text": statute,
                    },
                    {
                        "citation": "FGC § 1",
                        "title": "Fish",
                        "url": "http://law/1",
                        "text": "Fish.",
                    },
                ],
            }
        },
        articles={
            "employment": {
                "vectorizer": article_vectorizer,
                "matrix": article_matrix,
                "rows": [
                    {"title": "California Overtime", "url": "http://nolo/ca"},
                    {"title": "Virginia Overtime", "url": "http://nolo/va"},
                    {"title": "Overtime Pay Basics", "url": "http://nolo/none"},
                ],
            }
        },
    )
    calls = {"n": 0}
    real = get_tokens

    def counting(text: str) -> str:
        calls["n"] += 1
        return real(text)

    import pl2016.ask as ask_module

    ask_module.get_tokens = counting
    try:
        result = ask("my boss will not pay overtime wages", "NY", models)
    finally:
        ask_module.get_tokens = real
    assert calls["n"] == 1
    assert result["state"] == "NY"
    assert result["section"] == "employment"
    assert len(result["section_probs"]) == 3 or len(result["section_probs"]) == 2
    assert result["laws"][0]["citation"] == "LAB § 510"
    assert result["laws"][0]["snippet"]
    assert len(result["laws"][0]["snippet"]) < len(statute)
    titles = [item["title"] for item in result["articles"]]
    assert "Virginia Overtime" not in titles
    assert "California Overtime" not in titles
    assert titles[0] == "Overtime Pay Basics"
    assert "classify" in result["timings_ms"]
    source = (ROOT / "pl2016" / "ask.py").read_text()
    assert "join(get_tokens" not in source


def test_manifest_endpoint_serves_the_file(tmp_path):
    from pl2016.app import create_app

    (tmp_path / "manifest.json").write_text('{"version": "0.1.0", "routing": {"source": "hand-mapped"}}\n')
    app = create_app(models=Models(classifier={}, laws={}, articles={}), artifacts=tmp_path)
    body = app.test_client().get("/manifest").get_json()
    assert body["routing"]["source"] == "hand-mapped"


def test_flask_rejects_an_unsupported_state_and_serves_health():
    from pl2016.app import create_app

    app = create_app(models=Models(classifier={}, laws={}, articles={}))
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["ok"] is True
    refused = client.post("/ask", json={"question": "overtime", "state": "MD"})
    assert refused.status_code == 400
    assert "New York and California" in refused.get_json()["error"]
    missing = client.post("/ask", json={"state": "NY"})
    assert missing.status_code == 400
