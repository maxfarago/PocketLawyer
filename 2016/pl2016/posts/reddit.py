"""r/legaladvice posts from Li et al. 2022. This is a loader, not a scraper.

FindLaw Answers, the 2016 label source, is gone. Flair on these posts plays the
role the forum section played: the category attached to a lay question.

Dataset: jonathanli/legal-advice-reddit
Paper: https://aclanthology.org/2022.nllp-1.10

The published train/validation/test split is a few-shot split (the test file
holds most of the rows). Pool all three splits and draw our own stratified split
at train time.

    python -m pl2016.posts.reddit
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

DATASET_ID = "jonathanli/legal-advice-reddit"
SPLITS = ("train", "validation", "test")
CITATION = (
    "Li, Jonathan, Rohan Bhambhoria, and Xiaodan Zhu. 2022. "
    "Parameter-Efficient Legal Domain Adaptation. NLLP. "
    "https://aclanthology.org/2022.nllp-1.10"
)

# Column names confirmed against train.jsonl.
FIELDS = (
    "id",
    "title",
    "body",
    "text_label",
    "flair_label",
    "full_link",
    "created_utc",
)

LABELS = (
    "business",
    "contract",
    "criminal",
    "digital",
    "driving",
    "employment",
    "family",
    "housing",
    "insurance",
    "school",
    "wills",
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "reddit"
REPORT_PATH = ROOT / "artifacts" / "posts_report.json"
MONGO_URI = "mongodb://127.0.0.1:27017"
BATCH = 1000


def row_to_doc(row: dict, loaded_at: str) -> dict | None:
    """Shape one JSONL row into a POSTS document. None means skip."""
    reddit_id = (row.get("id") or "").strip()
    section = (row.get("text_label") or "").strip()
    title = (row.get("title") or "").strip()
    body = (row.get("body") or "").strip()
    if not reddit_id or not section or not (title or body):
        return None
    text = f"{title}\n{body}".strip()
    return {
        "id": reddit_id,
        "title": title,
        "text": text,
        "url": row.get("full_link") or "",
        "section": section,
        "created_utc": row.get("created_utc"),
        "flair_label": row.get("flair_label"),
        "loaded_at": loaded_at,
    }


def iter_rows(directory: Path):
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield split, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{line_number} is not valid JSON") from exc


def download(directory: Path = RAW_DIR) -> dict:
    """Save the three JSONL files and a provenance record. Returns the record."""
    from huggingface_hub import HfApi, hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(DATASET_ID)
    revision = info.sha
    for split in SPLITS:
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=f"{split}.jsonl",
            repo_type="dataset",
            revision=revision,
            local_dir=directory,
        )
    record = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "downloaded_at": _now(),
        "citation": CITATION,
        "files": [f"{split}.jsonl" for split in SPLITS],
    }
    (directory / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_posts(directory: Path = RAW_DIR, uri: str = MONGO_URI) -> dict:
    """Upsert pooled posts into POSTS.posts. A second run of the same rows changes nothing."""
    from pymongo import MongoClient, UpdateOne

    from pl2016.config import POSTS_COLLECTION, POSTS_DB

    loaded_at = _now()
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    collection = client[POSTS_DB][POSTS_COLLECTION]
    collection.create_index("id", unique=True)
    collection.create_index("section")

    counts: Counter[str] = Counter()
    skipped = 0
    unknown: Counter[str] = Counter()
    batch: list = []
    seen: set[str] = set()

    def flush() -> None:
        if batch:
            collection.bulk_write(batch, ordered=False)
            batch.clear()

    for _split, row in iter_rows(directory):
        doc = row_to_doc(row, loaded_at)
        if doc is None:
            skipped += 1
            continue
        if doc["id"] in seen:
            continue
        seen.add(doc["id"])
        if doc["section"] not in LABELS:
            unknown[doc["section"]] += 1
        counts[doc["section"]] += 1
        fields = {key: value for key, value in doc.items() if key != "loaded_at"}
        batch.append(
            UpdateOne(
                {"id": doc["id"]},
                {"$set": fields, "$setOnInsert": {"loaded_at": loaded_at}},
                upsert=True,
            )
        )
        if len(batch) >= BATCH:
            flush()
    flush()
    report = _report(collection, skipped, unknown)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _report(collection, skipped: int, unknown: Counter[str]) -> dict:
    lengths: list[int] = []
    empty_bodies = 0
    per_section: Counter[str] = Counter()
    for doc in collection.find({}, {"section": 1, "title": 1, "text": 1}):
        per_section[doc.get("section") or ""] += 1
        text = doc.get("text") or ""
        title = doc.get("title") or ""
        lengths.append(len(text))
        body = text[len(title):].strip() if text.startswith(title) else text
        if not body:
            empty_bodies += 1
    return {
        "documents": sum(per_section.values()),
        "skipped_rows": skipped,
        "unknown_labels": dict(unknown),
        "empty_bodies": empty_bodies,
        "median_text_length": int(median(lengths)) if lengths else 0,
        "per_section": dict(sorted(per_section.items())),
    }


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> None:
    print(f"downloading {DATASET_ID}")
    provenance = download()
    print(f"revision {provenance['revision']}")
    report = load_posts()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

