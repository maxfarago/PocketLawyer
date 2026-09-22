import json
from pathlib import Path

import pytest

from pl2016.train import (
    TOKEN_COLLECTIONS,
    build_manifest,
    evaluate_posts,
    fit_article_route,
    fit_law_route,
    fit_law_route_or_skip,
    make_post_vectorizer,
    matching_articles,
    matching_laws,
    posts_to_xy,
    slug,
)

ROOT = Path(__file__).resolve().parents[1]


def test_tokenization_targets_only_the_three_corpora():
    assert TOKEN_COLLECTIONS == (
        ("POSTS", "posts"),
        ("ARTICLES", "articles"),
        ("LAWS", "NY"),
        ("LAWS", "CA"),
    )
    source = (ROOT / "pl2016" / "train.py").read_text()
    assert "list_database_names" not in source
    assert "database_names" not in source
    assert "join(get_tokens" not in source


def test_labels_come_from_the_same_pass_as_the_tokens():
    docs = [
        {"id": "b", "tokens": "lease", "section": "housing"},
        {"id": "a", "tokens": "wage", "section": "employment"},
    ]
    ids, tokens, labels = posts_to_xy(docs)
    assert ids == ["b", "a"]
    assert tokens == ["lease", "wage"]
    assert labels == ["housing", "employment"]


def test_character_join_empties_the_vocabulary():
    from pl2016.text import get_tokens

    tokens = get_tokens("Unpaid overtime after the shift")
    vectorizer = make_post_vectorizer()
    fitted = vectorizer.fit_transform([tokens])
    assert fitted.shape[1] > 0
    broken = make_post_vectorizer()
    with pytest.raises(ValueError, match="empty vocabulary"):
        broken.fit_transform([" ".join(tokens)])


def test_slug_strips_every_character_the_notebook_reset():
    name = "a/b&c\\d,e.f:g h"
    broken = name
    for char in (" ", "/", "&", "\\", ",", ".", ":"):
        broken = name.replace(char, "")
    assert broken == "a/b&c\\d,e.fg h"
    assert slug(name) == "abcdefgh"


def test_law_prefix_keeps_descendants_and_drops_a_bare_title():
    docs = [
        {"url": "a", "section": ["Labor Code - LAB", "DIVISION 2. EMPLOYMENT", "Section 510"]},
        {"url": "b", "section": ["Labor Code - LAB", "DIVISION 4. WORKERS", "Section 3200"]},
        {"url": "c", "section": ["Penal Code - PEN", "DIVISION 2. EMPLOYMENT"]},
        {"url": "a", "section": ["Labor Code - LAB", "DIVISION 2. EMPLOYMENT", "Section 510"]},
    ]
    matched = matching_laws(docs, [["Labor Code - LAB", "DIVISION 2. EMPLOYMENT"]])
    assert [doc["url"] for doc in matched] == ["a"]


def test_empty_law_route_is_skipped_instead_of_raising():
    assert fit_law_route_or_skip([]) is None
    empty_vocabulary = {
        "tokens": "",
        "citation": "X § 1",
        "title": "Empty",
        "url": "http://example.test/1",
        "text": "",
        "section": ["Labor Code - LAB"],
    }
    assert fit_law_route_or_skip([empty_vocabulary]) is None


def test_manifest_records_counts_provenance_and_metrics(tmp_path):
    (tmp_path / "posts_metrics.json").write_text(
        json.dumps(
            {
                "trained_at": "2026-09-22T00:00:00+00:00",
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "versions": {"nltk": "3.10.3"},
                "tokens_written": {"LAWS.CA": 10},
                "per_flair": {},
            }
        )
    )
    (tmp_path / "posts_report.json").write_text(json.dumps({"documents": 3}))
    (tmp_path / "laws_report.json").write_text(
        json.dumps(
            {
                "trained_at": "2026-09-22T01:00:00+00:00",
                "tokens_written": {"LAWS.NY": 2},
                "routes": {"employment": {"NY": {"documents": 2}, "CA": {"documents": 4}}},
                "skipped": [],
            }
        )
    )
    (tmp_path / "articles_report.json").write_text(
        json.dumps(
            {
                "trained_at": "2026-09-22T02:00:00+00:00",
                "routes": {"employment": {"documents": 5}},
                "skipped": [{"flair": "school"}],
            }
        )
    )
    (tmp_path / "nolo_audit.json").write_text(
        json.dumps({"documents": 7, "empty_text": 0, "missing_area": 1})
    )
    manifest = build_manifest(tmp_path)
    assert manifest["routing"]["source"] == "hand-mapped"
    assert manifest["counts"]["posts"] == 3
    assert manifest["counts"]["articles"] == 7
    assert manifest["counts"]["laws"] == {"NY": 2, "CA": 10}
    assert manifest["law_documents"]["employment"]["CA"] == 4
    assert manifest["article_documents"]["employment"] == 5
    assert manifest["metrics"]["accuracy"] == 0.5
    assert manifest["dates"]["routes_mapped_at"] == "2026-09-22"


def test_empty_law_route_matches_nothing():
    docs = [{"url": "a", "section": ["Labor Code - LAB"]}]
    assert matching_laws(docs, []) == []


def test_articles_use_mapped_areas_only():
    docs = [
        {"url": "https://nolo.test/contract", "area": "Small Claims Court & Lawsuits", "title": "Sue"},
        {"url": "https://nolo.test/employment", "area": "Employment Law", "title": "Wage"},
        {"url": "https://nolo.test/laws", "area": "laws", "title": "Key name"},
    ]
    matched = matching_articles(docs, ["Small Claims Court & Lawsuits"])
    assert [doc["url"] for doc in matched] == ["https://nolo.test/contract"]
    assert matching_articles(docs, []) == []


def test_article_matrix_rows_match_the_display_rows():
    rows = [
        {"tokens": "breach contract", "title": "Contracts 101", "url": "https://nolo.test/a", "area": "Business Formation: LLCs & Corporations"},
        {"tokens": "small claim suit", "title": "Small Claims", "url": "https://nolo.test/b", "area": "Small Claims Court & Lawsuits"},
    ]
    _vectorizer, matrix, display = fit_article_route(rows)
    assert matrix.shape[0] == len(display) == 2
    assert display[0]["title"] == "Contracts 101"
    assert "text" not in display[0]


def test_law_matrix_rows_match_the_display_rows():
    rows = [
        {
            "tokens": "overtim wage",
            "citation": "LAB § 510",
            "title": "Section 510",
            "url": "https://example.test/510",
            "text": "Eight hours.",
            "section": ["Labor Code - LAB"],
        },
        {
            "tokens": "minimum wage",
            "citation": "LAB § 1182",
            "title": "Section 1182",
            "url": "https://example.test/1182",
            "text": "The commission.",
            "section": ["Labor Code - LAB"],
        },
    ]
    _vectorizer, matrix, display = fit_law_route(rows)
    assert matrix.shape[0] == len(display) == 2
    assert display[0]["citation"] == "LAB § 510"
    assert display[1]["url"].endswith("/1182")


def test_held_out_metrics_cover_every_flair():
    from pl2016.posts.reddit import LABELS

    tokens = []
    labels = []
    for label in LABELS:
        for index in range(5):
            tokens.append(f"{label} question number {index}")
            labels.append(label)
    report = evaluate_posts(tokens, labels)
    assert set(report["per_flair"]) == set(LABELS)
    assert report["confusion_matrix"]["labels"] == list(LABELS)
    assert report["random_state"] == 2016
    assert len(report["confusion_matrix"]["matrix"]) == len(LABELS)
