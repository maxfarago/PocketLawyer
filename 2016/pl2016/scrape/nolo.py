"""Nolo legal encyclopedia -> ARTICLES.

Article pages carry schema.org JSON-LD (headline, articleBody, breadcrumb).
The sitemap is the URL list. Listing pages are JavaScript shells, so we do not crawl them.

    python -m pl2016.scrape.nolo
    python -m pl2016.scrape.nolo --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from pl2016.scrape.cache import read_html, write_html

USER_AGENT = "PocketLawyer/2016 (educational rebuild; contact m@maxfarago.com)"
SITEMAP_URL = "https://www.nolo.com/sitemaps/nolo.com/sitemap.xml"
ROBOTS_URL = "https://www.nolo.com/robots.txt"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "nolo"
REPORT_PATH = ROOT / "artifacts" / "nolo_audit.json"
MONGO_URI = "mongodb://127.0.0.1:27017"
DEFAULT_DELAY = 0.75


def encyclopedia_urls(sitemap_xml: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for loc in re.findall(r"<loc>(.*?)</loc>", sitemap_xml):
        url = unescape(loc.strip())
        if "/legal-encyclopedia/" not in url or not url.endswith(".html"):
            continue
        if url in seen:
            continue
        seen.add(url)
        found.append(url)
    return found


def parse_article(html: str, url: str) -> dict | None:
    """Pull title, text, and the level-2 breadcrumb area from JSON-LD. None if this is not an article."""
    article = None
    crumbs: list[tuple[int, str, str]] = []
    for block in re.findall(
        r'<script type="application/ld\+json">(.*?)</script>', html, flags=re.S
    ):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        nodes = data.get("@graph") if isinstance(data, dict) else None
        if nodes is None and isinstance(data, dict):
            nodes = [data]
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            kind = node.get("@type")
            if kind == "Article" or (isinstance(kind, list) and "Article" in kind):
                article = node
            if kind == "BreadcrumbList":
                crumbs.extend(_crumbs(node))
    if article is None:
        return None
    crumbs.sort()
    area = ""
    area_url = ""
    for position, name, crumb_url in crumbs:
        if position == 2 and name:
            area = name
            area_url = crumb_url
            break
    title = (article.get("headline") or article.get("name") or "").strip()
    text = (article.get("articleBody") or "").strip()
    if not title and not text:
        return None
    return {
        "url": url,
        "title": title,
        "text": text,
        "area": area,
        "area_url": area_url,
    }


def _crumbs(node: dict) -> list[tuple[int, str, str]]:
    found = []
    for element in node.get("itemListElement") or []:
        if not isinstance(element, dict):
            continue
        item = element.get("item") or {}
        if isinstance(item, str):
            name, crumb_url = "", item
        else:
            name = (item.get("name") or element.get("name") or "").strip()
            crumb_url = item.get("@id") or item.get("id") or ""
        position = element.get("position") or 0
        found.append((int(position), name, crumb_url))
    return found


def allowed(robots: RobotFileParser, url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc not in ("www.nolo.com", "nolo.com"):
        return False
    return robots.can_fetch(USER_AGENT, url)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fetch(url: str, timeout: int = 30) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _load_done(path: Path) -> dict[str, int]:
    done: dict[str, int] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done[row["url"]] = row["status"]
    return done


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def crawl(limit: int | None = None, delay: float = DEFAULT_DELAY, uri: str = MONGO_URI) -> dict:
    from pymongo import MongoClient, UpdateOne

    from pl2016.config import ARTICLES_COLLECTION, ARTICLES_DB

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    robots_path = RAW_DIR / "robots.txt"
    robots_status, robots_body = _fetch_with_backoff(ROBOTS_URL)
    if robots_status == 200 and robots_body:
        robots_path.write_bytes(robots_body)
    elif robots_path.exists():
        robots_body = robots_path.read_bytes()
    else:
        raise RuntimeError(f"robots.txt returned {robots_status}")
    robots = RobotFileParser()
    robots.parse(robots_body.decode("utf-8", "replace").splitlines())
    sitemap_path = RAW_DIR / "sitemap.xml"
    if sitemap_path.exists():
        sitemap = sitemap_path.read_text(encoding="utf-8")
    else:
        status, body = _fetch(SITEMAP_URL)
        if status != 200:
            raise RuntimeError(f"sitemap returned {status}")
        sitemap_path.write_bytes(body)
        sitemap = body.decode("utf-8", "replace")

    urls = [url for url in encyclopedia_urls(sitemap) if allowed(robots, url)]
    if limit is not None:
        urls = urls[:limit]

    manifest_path = RAW_DIR / "manifest.jsonl"
    failures_path = RAW_DIR / "failures.jsonl"
    done = _load_done(manifest_path)

    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    collection = client[ARTICLES_DB][ARTICLES_COLLECTION]
    collection.create_index("url", unique=True)
    collection.create_index("area")

    fetched = cached = articles = skipped = failed = 0
    for index, url in enumerate(urls, start=1):
        status = done.get(url)
        if status is None:
            status, body = _fetch_with_backoff(url)
            if status == 200 and body:
                write_html(RAW_DIR, url, body)
                _append_jsonl(
                    manifest_path,
                    {"url": url, "status": status, "fetched_at": _now()},
                )
                fetched += 1
            else:
                _append_jsonl(failures_path, {"url": url, "status": status, "fetched_at": _now()})
                # Status 0 is a timeout. Leave it out of the manifest so a later run retries it.
                if status not in (0, 200):
                    _append_jsonl(
                        manifest_path,
                        {"url": url, "status": status, "fetched_at": _now()},
                    )
                failed += 1
                status = 0 if status == 200 else status
            time.sleep(delay)
        else:
            cached += 1

        if status != 200:
            continue
        raw = read_html(RAW_DIR, url)
        if raw is None:
            continue
        doc = parse_article(raw.decode("utf-8", "replace"), url)
        if doc is None:
            skipped += 1
            continue
        fields = {key: value for key, value in doc.items()}
        collection.bulk_write(
            [
                UpdateOne(
                    {"url": doc["url"]},
                    {"$set": fields, "$setOnInsert": {"scraped_at": _now()}},
                    upsert=True,
                )
            ]
        )
        articles += 1
        if index % 25 == 0 or index == len(urls):
            print(
                f"{index}/{len(urls)} fetched={fetched} cached={cached} "
                f"articles={articles} skipped={skipped} failed={failed}",
                flush=True,
            )

    report = _audit(collection)
    report["run"] = {
        "considered": len(urls),
        "fetched": fetched,
        "cached": cached,
        "upserted_this_run": articles,
        "not_articles": skipped,
        "failed": failed,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["run"], indent=2), flush=True)
    return report


def _fetch_with_backoff(url: str) -> tuple[int, bytes]:
    wait = 2.0
    status, body = 0, b""
    for _attempt in range(4):
        try:
            status, body = _fetch(url)
        except (TimeoutError, urllib.error.URLError):
            status, body = 0, b""
        if status not in (0, 429, 500, 502, 503, 504):
            return status, body
        time.sleep(wait)
        wait *= 2
    return status, body


def _audit(collection) -> dict:
    per_area: dict[str, int] = {}
    empty_text = 0
    missing_area = 0
    total = 0
    for doc in collection.find({}, {"area": 1, "text": 1, "url": 1}):
        total += 1
        area = doc.get("area") or ""
        per_area[area] = per_area.get(area, 0) + 1
        if not (doc.get("text") or "").strip():
            empty_text += 1
        if not area:
            missing_area += 1
    return {
        "documents": total,
        "empty_text": empty_text,
        "missing_area": missing_area,
        "per_area": dict(sorted(per_area.items(), key=lambda item: (-item[1], item[0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    args = parser.parse_args()
    crawl(limit=args.limit, delay=args.delay)


if __name__ == "__main__":
    main()
