"""Fit the post classifier, the law rankers, and the article rankers.

Token fields are written only on POSTS, ARTICLES, and LAWS.

    python -m pl2016.train
    python -m pl2016.train laws
    python -m pl2016.train articles
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pl2016.config import (
    ARTICLES_COLLECTION,
    ARTICLES_DB,
    LAWS_DB,
    POSTS_COLLECTION,
    POSTS_DB,
    STATES,
)
from pl2016.posts.reddit import LABELS
from pl2016.text import get_tokens

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CLASSIFIER_PATH = ARTIFACTS / "posts_classifier.joblib"
METRICS_PATH = ARTIFACTS / "posts_metrics.json"
MONGO_URI = "mongodb://127.0.0.1:27017"

# These four collections, and no others. The notebook walked every database.
TOKEN_COLLECTIONS = (
    (POSTS_DB, POSTS_COLLECTION),
    (ARTICLES_DB, ARTICLES_COLLECTION),
    (LAWS_DB, STATES[0]),
    (LAWS_DB, STATES[1]),
)

POST_VECTORIZER_PARAMS = {
    "max_features": 10000,
    "lowercase": False,
    "norm": "l2",
    "token_pattern": r"(?u)\b\w\w+\b",
}
# The notebook did not cap the law vocabulary. Same token pattern, so a second
# join of a token string still produces an empty vocabulary instead of a matrix.
LAW_VECTORIZER_PARAMS = {
    "lowercase": False,
    "norm": "l2",
    "token_pattern": r"(?u)\b\w\w+\b",
}
SLUG_CHARS = (" ", "/", "&", "\\", ",", ".", ":")
SPLIT_TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 2016
BATCH = 1000


def make_post_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(**POST_VECTORIZER_PARAMS)


def make_law_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(**LAW_VECTORIZER_PARAMS)


def slug(name: str) -> str:
    """Strip the notebook's filename punctuation in one pass.

    The notebook assigned `filename = forumSection.replace(c, "")` on every
    character, so each replacement started over and only `:` survived.
    """
    cleaned = name
    for char in SLUG_CHARS:
        cleaned = cleaned.replace(char, "")
    return cleaned


def path_matches(section: list, prefix: list) -> bool:
    return list(section[: len(prefix)]) == list(prefix)


def matching_laws(docs, prefixes: list[list[str]]) -> list:
    """Documents whose section path starts with any prefix. Each document once."""
    if not prefixes:
        return []
    chosen = []
    seen: set = set()
    for doc in docs:
        key = doc.get("url") or doc.get("_id")
        if key in seen:
            continue
        section = doc.get("section") or []
        if any(path_matches(section, prefix) for prefix in prefixes):
            seen.add(key)
            chosen.append(doc)
    return chosen


def posts_to_xy(docs) -> tuple[list[str], list[str], list[str]]:
    """One pass. Row i of the matrix is document i."""
    ids: list[str] = []
    tokens: list[str] = []
    labels: list[str] = []
    for doc in docs:
        ids.append(doc["id"])
        tokens.append(doc.get("tokens") or "")
        labels.append(doc["section"])
    return ids, tokens, labels


def evaluate_posts(tokens: list[str], labels: list[str]) -> dict:
    """Held-out metrics. The vectorizer is fit on the train split only."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    text_train, text_test, y_train, y_test = train_test_split(
        tokens,
        labels,
        test_size=SPLIT_TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=labels,
    )
    vectorizer = make_post_vectorizer()
    classifier = MultinomialNB()
    classifier.fit(vectorizer.fit_transform(text_train), y_train)
    predicted = classifier.predict(vectorizer.transform(text_test))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=list(LABELS), zero_division=0
    )
    matrix = confusion_matrix(y_test, predicted, labels=list(LABELS))
    return {
        "test_size": SPLIT_TEST_SIZE,
        "random_state": SPLIT_RANDOM_STATE,
        "held_out_documents": len(y_test),
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "macro_f1": round(float(f1_score(y_test, predicted, average="macro")), 4),
        "per_flair": {
            label: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": matrix.tolist(),
        },
    }


def fit_posts(tokens: list[str], labels: list[str]):
    """Refit on every post. This is the model that serves."""
    from sklearn.naive_bayes import MultinomialNB

    vectorizer = make_post_vectorizer()
    matrix = vectorizer.fit_transform(tokens)
    classifier = MultinomialNB()
    classifier.fit(matrix, labels)
    return vectorizer, classifier, matrix


def write_tokens(
    uri: str = MONGO_URI,
    collections=TOKEN_COLLECTIONS,
    missing_only: bool = False,
) -> dict:
    from pymongo import MongoClient, UpdateOne

    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    written: dict[str, int] = {}
    for db_name, collection_name in collections:
        database = client[db_name]
        if collection_name not in database.list_collection_names():
            print(f"tokens skip {db_name}.{collection_name}", flush=True)
            written[f"{db_name}.{collection_name}"] = 0
            continue
        collection = database[collection_name]
        batch: list = []
        count = 0

        def flush() -> None:
            nonlocal count
            if not batch:
                return
            collection.bulk_write(batch, ordered=False)
            count += len(batch)
            batch.clear()
            print(f"tokens {db_name}.{collection_name} {count}", flush=True)

        query = {"tokens": {"$exists": False}} if missing_only else {}
        for doc in collection.find(query, {"text": 1}, batch_size=BATCH):
            batch.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {"tokens": get_tokens(doc.get("text") or "")}},
                )
            )
            if len(batch) >= BATCH:
                flush()
        flush()
        written[f"{db_name}.{collection_name}"] = count
    return written


def train(uri: str = MONGO_URI) -> dict:
    import joblib
    from pymongo import MongoClient

    written = write_tokens(uri)
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    docs = list(
        client[POSTS_DB][POSTS_COLLECTION]
        .find({}, {"id": 1, "tokens": 1, "section": 1})
        .sort("id", 1)
    )
    ids, tokens, labels = posts_to_xy(docs)
    metrics = evaluate_posts(tokens, labels)
    vectorizer, classifier, matrix = fit_posts(tokens, labels)
    if matrix.shape[0] != len(ids):
        raise RuntimeError("post matrix rows do not match the document list")
    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report = {
        "trained_at": trained_at,
        "documents": len(ids),
        "tokens_written": written,
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "vectorizer": POST_VECTORIZER_PARAMS,
        "classifier": "sklearn.naive_bayes.MultinomialNB defaults",
        "refit": "all posts, after held-out metrics",
        "versions": {
            "scikit-learn": version("scikit-learn"),
            "nltk": version("nltk"),
            "numpy": version("numpy"),
        },
        **metrics,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier, "ids": ids},
        CLASSIFIER_PATH,
    )
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    write_manifest()
    return report


def fit_law_route_or_skip(rows: list[dict]):
    """None when the route has no sections or the vocabulary is empty.

    The notebook fitted every section and crashed on an empty vocabulary.
    """
    if not rows:
        return None
    try:
        return fit_law_route(rows)
    except ValueError:
        return None


def fit_law_route(rows: list[dict]):
    """Vectorizer, matrix, and display rows. Row i is document i."""
    tokens = [row.get("tokens") or "" for row in rows]
    vectorizer = make_law_vectorizer()
    matrix = vectorizer.fit_transform(tokens)
    display = [
        {
            "citation": row.get("citation") or "",
            "title": row.get("title") or "",
            "url": row.get("url") or "",
            "text": row.get("text") or "",
            "section": row.get("section") or [],
        }
        for row in rows
    ]
    if matrix.shape[0] != len(display):
        raise RuntimeError("law matrix rows do not match the display rows")
    return vectorizer, matrix, display


def _load_routed_laws(collection, prefixes: list[list[str]]) -> list[dict]:
    if not prefixes:
        return []
    clauses = [
        {"$expr": {"$eq": [{"$slice": ["$section", len(prefix)]}, prefix]}} for prefix in prefixes
    ]
    found = collection.find(
        {"$or": clauses},
        {"citation": 1, "title": 1, "url": 1, "text": 1, "section": 1, "tokens": 1},
    ).sort("citation", 1)
    return matching_laws(found, prefixes)


def train_laws(uri: str = MONGO_URI) -> dict:
    import joblib
    from pymongo import MongoClient

    written = write_tokens(
        uri,
        collections=((LAWS_DB, STATES[0]), (LAWS_DB, STATES[1])),
        missing_only=True,
    )
    routes = json.loads((ROOT / "routing_tables" / "flair_routes.json").read_text(encoding="utf-8"))
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report: dict = {"trained_at": trained_at, "tokens_written": written, "routes": {}, "skipped": []}
    for label in LABELS:
        mapping = routes["mappings"][label]
        report["routes"][label] = {}
        for state in STATES:
            prefixes = mapping["laws"][state]
            rows = _load_routed_laws(client[LAWS_DB][state], prefixes)
            fitted = fit_law_route_or_skip(rows)
            if fitted is None:
                print(f"laws skip {state} {label}", flush=True)
                report["skipped"].append({"state": state, "flair": label, "reason": "no sections"})
                continue
            vectorizer, matrix, display = fitted
            path = ARTIFACTS / "laws" / state / f"{slug(label)}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"vectorizer": vectorizer, "matrix": matrix, "rows": display},
                path,
            )
            report["routes"][label][state] = {
                "documents": len(display),
                "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "file": str(path.relative_to(ROOT)),
            }
            print(f"laws {state} {label} {len(display)}", flush=True)
    out = ARTIFACTS / "laws_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    write_manifest()
    return report


def matching_articles(docs, areas: list[str]) -> list:
    """Articles in the mapped Nolo areas. Not every area, and not the mapping keys."""
    if not areas:
        return []
    wanted = set(areas)
    chosen = []
    seen: set = set()
    for doc in docs:
        key = doc.get("url") or doc.get("_id")
        if key in seen or doc.get("area") not in wanted:
            continue
        seen.add(key)
        chosen.append(doc)
    return chosen


def fit_article_route(rows: list[dict]):
    """Vectorizer, matrix, and display rows. The public fields are title and url."""
    tokens = [row.get("tokens") or "" for row in rows]
    vectorizer = make_law_vectorizer()
    matrix = vectorizer.fit_transform(tokens)
    display = [
        {
            "title": row.get("title") or "",
            "url": row.get("url") or "",
            "area": row.get("area") or "",
        }
        for row in rows
    ]
    if matrix.shape[0] != len(display):
        raise RuntimeError("article matrix rows do not match the display rows")
    return vectorizer, matrix, display


def train_articles(uri: str = MONGO_URI) -> dict:
    import joblib
    from pymongo import MongoClient

    from pl2016.routing.validate import article_areas

    written = write_tokens(
        uri,
        collections=((ARTICLES_DB, ARTICLES_COLLECTION),),
        missing_only=True,
    )
    routes = json.loads((ROOT / "routing_tables" / "flair_routes.json").read_text(encoding="utf-8"))
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    collection = client[ARTICLES_DB][ARTICLES_COLLECTION]
    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report: dict = {"trained_at": trained_at, "tokens_written": written, "routes": {}, "skipped": []}
    for label in LABELS:
        areas = article_areas(routes["mappings"][label])
        if not areas:
            print(f"articles skip {label}", flush=True)
            report["skipped"].append({"flair": label, "reason": "statutes only"})
            continue
        found = collection.find(
            {"area": {"$in": areas}},
            {"title": 1, "url": 1, "area": 1, "tokens": 1},
        ).sort("url", 1)
        rows = matching_articles(found, areas)
        if not rows:
            print(f"articles skip {label}: no documents", flush=True)
            report["skipped"].append({"flair": label, "reason": "no articles"})
            continue
        try:
            vectorizer, matrix, display = fit_article_route(rows)
        except ValueError as exc:
            print(f"articles skip {label}: {exc}", flush=True)
            report["skipped"].append({"flair": label, "reason": str(exc)})
            continue
        path = ARTIFACTS / "articles" / f"{slug(label)}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "rows": display}, path)
        report["routes"][label] = {
            "areas": areas,
            "documents": len(display),
            "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "file": str(path.relative_to(ROOT)),
        }
        print(f"articles {label} {len(display)}", flush=True)
    out = ARTIFACTS / "articles_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    write_manifest()
    return report


def build_manifest(directory: Path | None = None) -> dict:
    """One record of counts, dates, routing provenance, metrics, and library versions."""
    directory = directory or ARTIFACTS
    routes = json.loads((ROOT / "routing_tables" / "flair_routes.json").read_text(encoding="utf-8"))
    posts = _read_json(directory / "posts_metrics.json")
    posts_loaded = _read_json(directory / "posts_report.json")
    laws = _read_json(directory / "laws_report.json")
    articles = _read_json(directory / "articles_report.json")
    nolo = _read_json(directory / "nolo_audit.json")
    law_counts = {}
    for label, by_state in laws.get("routes", {}).items():
        law_counts[label] = {}
        for state in STATES:
            law_counts[label][state] = (by_state.get(state) or {}).get("documents", 0)
    article_counts = {
        label: info.get("documents", 0) for label, info in articles.get("routes", {}).items()
    }
    return {
        "version": _package_version(),
        "written_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "states": list(STATES),
        "versions": posts.get("versions", {}),
        "routing": routes.get("provenance", {}),
        "counts": {
            "posts": posts_loaded.get("documents", posts.get("documents")),
            "articles": nolo.get("documents"),
            "articles_empty_text": nolo.get("empty_text"),
            "articles_missing_area": nolo.get("missing_area"),
            "laws": {
                "NY": _corpus_count("LAWS.NY", laws, posts),
                "CA": _corpus_count("LAWS.CA", laws, posts),
            },
        },
        "dates": {
            "routes_mapped_at": (routes.get("provenance") or {}).get("mapped_at"),
            "posts_trained_at": posts.get("trained_at"),
            "laws_trained_at": laws.get("trained_at"),
            "articles_trained_at": articles.get("trained_at"),
        },
        "metrics": {
            "accuracy": posts.get("accuracy"),
            "macro_f1": posts.get("macro_f1"),
            "held_out_documents": posts.get("held_out_documents"),
            "per_flair": posts.get("per_flair", {}),
        },
        "law_documents": law_counts,
        "article_documents": article_counts,
        "skipped": {
            "laws": laws.get("skipped", []),
            "articles": articles.get("skipped", []),
        },
    }


def write_manifest(directory: Path | None = None) -> dict:
    directory = directory or ARTIFACTS
    manifest = build_manifest(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _package_version() -> str:
    from pl2016 import __version__

    return __version__


def _corpus_count(key: str, *reports: dict) -> int | None:
    """The loaded corpus size, not the last missing-only write (that one is often 0)."""
    numbers = []
    for report in reports:
        value = (report.get("tokens_written") or {}).get(key)
        if isinstance(value, int):
            numbers.append(value)
    return max(numbers) if numbers else None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("laws", "articles"):
        report = train_laws() if sys.argv[1] == "laws" else train_articles()
        print(
            json.dumps(
                {"skipped": report["skipped"], "flairs": list(report["routes"])},
                indent=2,
            )
        )
        return
    report = train()
    print(
        json.dumps(
            {
                "documents": report["documents"],
                "accuracy": report["accuracy"],
                "macro_f1": report["macro_f1"],
                "matrix_shape": report["matrix_shape"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
