"""Reject a routing table that points at strings the database does not contain.

Reads JSON. Does not eval the file.

    python -m pl2016.routing.validate
"""

from __future__ import annotations

import json
from pathlib import Path

from pl2016.posts.reddit import LABELS

ROOT = Path(__file__).resolve().parents[2]
ROUTES_PATH = ROOT / "routing_tables" / "flair_routes.json"
MONGO_URI = "mongodb://127.0.0.1:27017"


def article_areas(mapping: dict) -> list[str]:
    """The mapped Nolo areas. Not the mapping's keys."""
    return list(mapping["articles"])


def prefix_problem(prefix) -> str | None:
    if not isinstance(prefix, list) or not prefix:
        return "a prefix is a non-empty list, not a bare title"
    if not all(isinstance(part, str) and part.strip() for part in prefix):
        return "a prefix contains a blank part"
    if " - " not in prefix[0]:
        return "the first element must name a code, not a bare title"
    return None


def prefix_hits(prefix: list[str], paths: list[list[str]]) -> bool:
    width = len(prefix)
    return any(path[:width] == prefix for path in paths)


def check_routes(
    routes: dict,
    *,
    ca_paths: list[list[str]] | None,
    ny_loaded: bool,
    ny_paths: list[list[str]] | None = None,
    areas: set[str] | None = None,
    areas_complete: bool = False,
) -> tuple[list[str], list[str]]:
    """Return errors and warnings. An unknown statute prefix is an error.

    An empty statute list is articles-only and is a warning. An empty article
    list is statutes-only and is a warning. Article names are checked against
    the crawled areas.
    New York prefixes are structured-checked and, until LAWS.NY exists, not
    compared to an inventory.
    """
    errors: list[str] = []
    warnings: list[str] = []
    mappings = routes.get("mappings") or {}
    if set(mappings) != set(LABELS):
        missing = sorted(set(LABELS) - set(mappings))
        extra = sorted(set(mappings) - set(LABELS))
        if missing:
            errors.append(f"missing flairs: {', '.join(missing)}")
        if extra:
            errors.append(f"unknown flairs: {', '.join(extra)}")
    if not ny_loaded:
        warnings.append("LAWS.NY is not loaded; New York prefixes were not checked")
    for label in LABELS:
        mapping = mappings.get(label)
        if mapping is None:
            continue
        mapped = article_areas(mapping)
        if mapped == []:
            warnings.append(f"{label}: statutes only, no Nolo area")
        elif areas_complete and areas is not None:
            for area in mapped:
                if area not in areas:
                    errors.append(f"{label}: unknown article area {area}")
        for state, paths, loaded in (
            ("CA", ca_paths, ca_paths is not None),
            ("NY", ny_paths, ny_loaded),
        ):
            prefixes = mapping.get("laws", {}).get(state)
            if prefixes is None:
                errors.append(f"{label}: missing {state} statute list")
                continue
            if prefixes == []:
                warnings.append(f"{label}: {state} is articles only")
                continue
            for prefix in prefixes:
                problem = prefix_problem(prefix)
                if problem:
                    errors.append(f"{label}: {state}: {problem}: {prefix!r}")
                    continue
                if loaded and paths is not None and not prefix_hits(prefix, paths):
                    errors.append(f"{label}: {state}: prefix is not in the inventory: {prefix}")
    return errors, warnings


def validate_stored_routes(uri: str = MONGO_URI) -> tuple[list[str], list[str]]:
    from pymongo import MongoClient

    from pl2016.config import ARTICLES_COLLECTION, ARTICLES_DB, LAWS_DB

    routes = json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    laws = client[LAWS_DB]
    ny_loaded = "NY" in laws.list_collection_names() and laws["NY"].estimated_document_count() > 0
    ca_paths = _paths(laws["CA"]) if "CA" in laws.list_collection_names() else []
    ny_paths = _paths(laws["NY"]) if ny_loaded else None
    areas = set(client[ARTICLES_DB][ARTICLES_COLLECTION].distinct("area"))
    return check_routes(
        routes,
        ca_paths=ca_paths,
        ny_loaded=ny_loaded,
        ny_paths=ny_paths,
        areas=areas,
        areas_complete=True,
    )


def _paths(collection) -> list[list[str]]:
    return [doc["section"] for doc in collection.find({}, {"section": 1}) if doc.get("section")]


def main() -> None:
    errors, warnings = validate_stored_routes()
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}")
    if errors:
        raise SystemExit(1)
    print("routes ok")


if __name__ == "__main__":
    main()
