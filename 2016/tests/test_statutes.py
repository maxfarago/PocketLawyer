from pl2016.scrape.california import lob_to_text, parse_row, section_path
from pl2016.scrape.newyork import iter_sections, node_label, routed_law_ids, senate_text
from pl2016.text import get_tokens


def test_parse_row_unquotes_backticks_and_nulls():
    row = parse_row("`LAB`\t`Labor Code - LAB`\tNULL\t2025-11-27 01:00:22")
    assert row == ["LAB", "Labor Code - LAB", None, "2025-11-27 01:00:22"]


def test_section_path_uses_official_toc_headings():
    nodes = [
        {
            "division": "2.",
            "title": None,
            "part": None,
            "chapter": None,
            "article": None,
            "heading": "DIVISION 2. EMPLOYMENT REGULATION AND SUPERVISION",
            "active": True,
            "level": 1,
            "sequence": 1,
        },
        {
            "division": "2.",
            "title": None,
            "part": "2.",
            "chapter": None,
            "article": None,
            "heading": "PART 2. WORKING HOURS",
            "active": True,
            "level": 2,
            "sequence": 1,
        },
    ]
    path = section_path(
        "Labor Code - LAB",
        nodes,
        {"division": "2.", "title": None, "part": "2.", "chapter": "1.", "article": None},
    )
    assert path == [
        "Labor Code - LAB",
        "DIVISION 2. EMPLOYMENT REGULATION AND SUPERVISION",
        "PART 2. WORKING HOURS",
    ]


def test_lob_text_drops_markup():
    xml = '<caml:Content><p>(a)<span class="EnSpace"/>Eight hours.</p><p>(b)<span class="EnSpace"/>Overtime.</p></caml:Content>'
    assert lob_to_text(xml) == "(a) Eight hours.\n(b) Overtime."


def test_senate_text_decodes_literal_newlines():
    decoded = senate_text("covered by this\\narticle, not\\nless")
    assert decoded == "covered by this\narticle, not\nless"
    assert "\\n" not in decoded
    assert "thisnarticl" not in get_tokens(decoded)
    assert "articl" in get_tokens(decoded)


def test_new_york_section_path_includes_the_law_and_the_article():
    tree = {
        "result": {
            "lawVersion": {"lawId": "LAB"},
            "info": {"lawId": "LAB", "name": "Labor"},
            "documents": {
                "docType": "CHAPTER",
                "title": "Labor",
                "documents": {
                    "items": [
                        {
                            "docType": "ARTICLE",
                            "docLevelId": "6",
                            "title": "PAYMENT OF WAGES",
                            "locationId": "A6",
                            "documents": {
                                "items": [
                                    {
                                        "docType": "SECTION",
                                        "docLevelId": "190",
                                        "locationId": "190",
                                        "title": "Payment of wages",
                                        "text": "Every employer shall pay wages.",
                                        "documents": {"items": []},
                                    }
                                ]
                            },
                        }
                    ]
                },
            },
        }
    }
    sections = list(iter_sections(tree))
    assert sections[0]["section"] == [
        "LAB - Labor",
        "Article 6 - PAYMENT OF WAGES",
        "Section 190 - Payment of wages",
    ]
    assert node_label({"docType": "CHAPTER"}, "LAB", "Labor") == "LAB - Labor"
    assert "LAB" in routed_law_ids()
