import json
from pathlib import Path

from pl2016.posts.reddit import LABELS, row_to_doc

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "routing_tables" / "flair_routes.json"


def test_row_combines_title_and_body_once():
    doc = row_to_doc(
        {
            "id": "abc",
            "title": "Unpaid overtime",
            "body": "My boss will not pay me.",
            "text_label": "employment",
            "full_link": "https://www.reddit.com/r/legaladvice/comments/abc/",
            "created_utc": 1588353888,
            "flair_label": 5,
        },
        loaded_at="2026-09-22T00:00:00+00:00",
    )
    assert doc is not None
    assert doc["section"] == "employment"
    assert doc["text"] == "Unpaid overtime\nMy boss will not pay me."
    assert doc["url"].endswith("/abc/")


def test_row_skips_a_post_with_no_words():
    assert row_to_doc({"id": "abc", "title": "  ", "body": "", "text_label": "housing"}, "t") is None
    assert row_to_doc({"id": "", "title": "Hi", "body": "There", "text_label": "housing"}, "t") is None


def test_routes_cover_every_dataset_label():
    routes = json.loads(ROUTES.read_text())
    assert routes["provenance"]["mapped_at"] == "2026-09-22"
    assert routes["provenance"]["source"] == "hand-mapped"
    assert set(routes["mappings"]) == set(LABELS)
    for label, mapping in routes["mappings"].items():
        assert isinstance(mapping["articles"], list), label
        for state in ("NY", "CA"):
            prefixes = mapping["laws"][state]
            assert prefixes, label
            for prefix in prefixes:
                assert prefix and all(isinstance(part, str) and part.strip() for part in prefix)
