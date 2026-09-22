from pathlib import Path

from pl2016.routing.validate import article_areas, check_routes, prefix_problem


def test_routing_files_are_json_not_eval():
    root = Path(__file__).resolve().parents[1]
    for relative in ("pl2016/routing/validate.py", "pl2016/train.py", "pl2016/ask.py"):
        source = (root / relative).read_text()
        assert "eval(" not in source
    assert "json.loads" in (root / "pl2016/routing/validate.py").read_text()


def test_areas_are_the_mapped_list_not_the_mapping_keys():
    mapping = {"articles": ["Employment Law"], "laws": {"NY": [], "CA": []}}
    assert article_areas(mapping) == ["Employment Law"]
    assert article_areas(mapping) != list(mapping)


def test_bare_title_is_rejected():
    assert prefix_problem("Title 2") is not None
    assert prefix_problem(["Title 2"]) is not None
    assert prefix_problem(["Penal Code - PEN"]) is None


def test_missing_prefix_is_an_error_and_an_empty_list_is_articles_only():
    from pl2016.posts.reddit import LABELS

    routes = {"mappings": {}}
    for label in LABELS:
        routes["mappings"][label] = {
            "articles": ["Employment Law"],
            "laws": {
                "NY": [["LAB - Labor"]],
                "CA": [["Labor Code - LAB", "DIVISION 2. EMPLOYMENT"]],
            },
        }
    routes["mappings"]["employment"]["laws"]["NY"] = [
        ["LAB - Labor", "Article 6 - Not A Real Heading"]
    ]
    routes["mappings"]["wills"]["laws"]["CA"] = []
    errors, warnings = check_routes(
        routes,
        ca_paths=[["Labor Code - LAB", "DIVISION 2. EMPLOYMENT", "Section 510"]],
        ny_loaded=True,
        ny_paths=[["LAB - Labor", "Article 19 - Minimum Wage Act"]],
        areas={"Employment Law"},
        areas_complete=True,
    )
    assert any("wills: CA is articles only" in warning for warning in warnings)
    assert any("employment: NY: prefix is not in the inventory" in error for error in errors)
    assert not any("employment: CA:" in error for error in errors)


def test_empty_article_list_is_statutes_only():
    from pl2016.posts.reddit import LABELS

    routes = {"mappings": {}}
    for label in LABELS:
        routes["mappings"][label] = {
            "articles": ["Employment Law"],
            "laws": {"NY": [["LAB - Labor"]], "CA": [["Labor Code - LAB"]]},
        }
    routes["mappings"]["wills"]["articles"] = []
    errors, warnings = check_routes(
        routes,
        ca_paths=[["Labor Code - LAB"]],
        ny_loaded=True,
        ny_paths=[["LAB - Labor"]],
        areas_complete=True,
        areas={"Employment Law"},
    )
    assert any("wills: statutes only, no Nolo area" in warning for warning in warnings)
    assert not any("wills" in error for error in errors)
