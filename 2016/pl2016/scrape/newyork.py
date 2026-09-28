"""Load New York consolidated laws from the Senate Open Legislation API into LAWS.NY.

Only the law volumes named in the routing table are fetched. Each volume is one
request with full text, cached under data/raw/ny/.

    python -m pl2016.scrape.newyork
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pl2016.scrape.california import REPO

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ny"
ROUTES = ROOT / "routing_tables" / "flair_routes.json"
ENV_PATH = ROOT / ".env"
API = "https://legislation.nysenate.gov/api/3/laws"
MONGO_URI = "mongodb://127.0.0.1:27017"
USER_AGENT = "PocketLawyer/2016 (educational rebuild; contact m@maxfarago.com)"


def api_key() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("NY_SENATE_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if key:
                return key
    raise RuntimeError("NY_SENATE_API_KEY is missing from 2016/.env")


def routed_law_ids() -> list[str]:
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))
    found: list[str] = []
    seen: set[str] = set()
    for mapping in routes["mappings"].values():
        for prefix in mapping["laws"]["NY"]:
            law_id = prefix[0].split(" - ", 1)[0].strip()
            if law_id and law_id not in seen:
                seen.add(law_id)
                found.append(law_id)
    return found


def node_label(node: dict, law_id: str, law_name: str) -> str:
    doc_type = (node.get("docType") or "").upper()
    level = str(node.get("docLevelId") or "").strip()
    title = (node.get("title") or "").strip()
    if doc_type == "CHAPTER":
        return f"{law_id} - {law_name}"
    kind = doc_type.title()
    if level and title:
        return f"{kind} {level} - {title}"
    return title or f"{kind} {level}".strip()


def senate_text(text: str) -> str:
    """The API stores line breaks as the two characters backslash and n."""
    return (text or "").replace("\\n", "\n").strip()


def iter_sections(tree: dict):
    result = tree.get("result") or tree
    info = result.get("info") or {}
    law_id = (result.get("lawVersion") or {}).get("lawId") or info.get("lawId")
    law_name = info.get("name") or law_id
    root = result.get("documents") or {}

    def walk(node: dict, ancestors: list[str]) -> None:
        if not isinstance(node, dict) or node.get("repealed"):
            return
        label = node_label(node, law_id, law_name)
        path = ancestors
        if label and (not path or path[-1] != label):
            path = [*ancestors, label] if (node.get("docType") or "").upper() != "CHAPTER" else (
                ancestors or [label]
            )
            if (node.get("docType") or "").upper() == "CHAPTER":
                path = [label]
        if (node.get("docType") or "").upper() == "SECTION":
            yield_path = path if path and path[-1] == label else [*path, label]
            yield {
                "law_id": law_id,
                "location_id": node.get("locationId"),
                "title": (node.get("title") or "").strip(),
                "text": senate_text(node.get("text") or ""),
                "section": yield_path,
            }
        documents = node.get("documents") or {}
        children = documents.get("items") if isinstance(documents, dict) else documents
        for child in children or []:
            yield from walk(child, path)

    yield from walk(root, [])


def fetch_law(law_id: str, key: str) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{law_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    query = urllib.parse.urlencode({"full": "true", "key": key})
    request = urllib.request.Request(
        f"{API}/{law_id}?{query}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    if not payload.get("success"):
        raise RuntimeError(payload.get("message") or f"{law_id} request failed")
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def repair_stored_text(uri: str = MONGO_URI) -> dict:
    """Rewrite LAWS.NY text that still contains a literal backslash-n."""
    from pymongo import MongoClient, UpdateOne

    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    collection = client["LAWS"]["NY"]
    batch: list = []
    rewritten = 0

    def flush() -> None:
        nonlocal rewritten
        if not batch:
            return
        collection.bulk_write(batch, ordered=False)
        rewritten += len(batch)
        batch.clear()
        print(f"new york text {rewritten}", flush=True)

    for doc in collection.find({"text": {"$regex": r"\\n"}}, {"text": 1}, batch_size=500):
        batch.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"text": senate_text(doc["text"])}}))
        if len(batch) >= 500:
            flush()
    flush()
    print(f"new york text done {rewritten}", flush=True)
    return {"rewritten": rewritten}


def load(uri: str = MONGO_URI) -> dict:
    from pymongo import MongoClient, UpdateOne

    key = api_key()
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    collection = client["LAWS"]["NY"]
    collection.create_index("url", unique=True)
    loaded_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    written = 0
    for law_id in routed_law_ids():
        print(f"new york {law_id}", flush=True)
        tree = fetch_law(law_id, key)
        batch = []
        for section in iter_sections(tree):
            location = section["location_id"]
            doc = {
                "state": "NY",
                "code": section["law_id"],
                "citation": f"{section['law_id']} § {location}",
                "title": section["title"],
                "url": f"https://www.nysenate.gov/legislation/laws/{section['law_id']}/{location}",
                "text": section["text"],
                "section": section["section"],
            }
            batch.append(
                UpdateOne(
                    {"url": doc["url"]},
                    {"$set": doc, "$setOnInsert": {"loaded_at": loaded_at}},
                    upsert=True,
                )
            )
        if batch:
            collection.bulk_write(batch, ordered=False)
            written += len(batch)
        print(f"new york {law_id} {len(batch)}", flush=True)
    print(f"new york done {written}", flush=True)
    return {"sections": written}


def main() -> None:
    load()


if __name__ == "__main__":
    main()
