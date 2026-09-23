"""Load California codes from the legislature's pubinfo zip into LAWS.CA.

The zip stays at the repo root and is gitignored. Section text is in the
LOB files; the hierarchy is LAW_TOC_TBL.

    python -m pl2016.scrape.california
"""

from __future__ import annotations

import argparse
import re
import zipfile
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ZIP = REPO / "pubinfo_2025.zip"
MONGO_URI = "mongodb://127.0.0.1:27017"

STRUCTURAL = ("division", "title", "part", "chapter", "article")


def parse_row(line: str) -> list[str | None]:
    fields = []
    for field in line.rstrip("\n").split("\t"):
        if field == "NULL" or field == "":
            fields.append(None)
        elif len(field) >= 2 and field[0] == "`" and field[-1] == "`":
            fields.append(field[1:-1])
        else:
            fields.append(field)
    return fields


def lob_to_text(xml: str) -> str:
    xml = xml.replace('<span class="EnSpace"/>', " ")
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", xml, flags=re.S)
    chunks = paragraphs or [xml]
    text = "\n".join(re.sub(r"<[^>]+>", "", chunk) for chunk in chunks)
    return unescape(text).strip()


def load_codes(archive: zipfile.ZipFile) -> dict[str, str]:
    raw = archive.read("CODES_TBL.dat").decode("utf-8", "replace")
    codes = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = parse_row(line)
        codes[row[0]] = row[1]
    return codes


def load_toc(archive: zipfile.ZipFile) -> dict[str, list[dict]]:
    raw = archive.read("LAW_TOC_TBL.dat").decode("utf-8", "replace")
    by_code: dict[str, list[dict]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = parse_row(line)
        node = {
            "division": row[1],
            "title": row[2],
            "part": row[3],
            "chapter": row[4],
            "article": row[5],
            "heading": (row[6] or "").strip(),
            "active": row[7] == "Y",
            "level": int(row[11] or 0),
            "sequence": int(row[12] or 0),
        }
        if not node["active"] or not any(node[key] for key in STRUCTURAL):
            continue
        by_code.setdefault(row[0], []).append(node)
    return by_code


def section_path(code_name: str, nodes: list[dict], section: dict) -> list[str]:
    hits = [node for node in nodes if _covers(node, section)]
    hits.sort(key=lambda node: (node["level"], node["sequence"]))
    headings: list[str] = []
    for node in hits:
        if node["heading"] and (not headings or headings[-1] != node["heading"]):
            headings.append(node["heading"])
    return [code_name, *headings]


def _covers(node: dict, section: dict) -> bool:
    for key in STRUCTURAL:
        if node[key] is not None and node[key] != section.get(key):
            return False
    return True


def section_url(code: str, number: str) -> str:
    return (
        "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml"
        f"?lawCode={code}&sectionNum={number.rstrip('.')}"
    )


def iter_sections(archive: zipfile.ZipFile):
    with archive.open("LAW_SECTION_TBL.dat") as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8", "replace")
            if not line.strip():
                continue
            row = parse_row(line)
            if row[15] != "Y" or not row[14]:
                continue
            yield {
                "code": row[1],
                "number": row[2],
                "division": row[8],
                "title": row[9],
                "part": row[10],
                "chapter": row[11],
                "article": row[12],
                "lob": row[14],
                "updated": row[17] or "",
            }


def load(zip_path: Path = DEFAULT_ZIP, uri: str = MONGO_URI) -> dict:
    from pymongo import MongoClient, UpdateOne

    archive = zipfile.ZipFile(zip_path)
    codes = load_codes(archive)
    toc = load_toc(archive)
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    collection = client["LAWS"]["CA"]
    collection.create_index("url", unique=True)

    loaded_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    chosen: dict[tuple[str, str], dict] = {}
    for section in iter_sections(archive):
        key = (section["code"], section["number"])
        previous = chosen.get(key)
        if previous is None or section["updated"] >= previous["updated"]:
            chosen[key] = section

    batch: list = []
    written = 0

    def flush() -> None:
        nonlocal written
        if not batch:
            return
        collection.bulk_write(batch, ordered=False)
        written += len(batch)
        batch.clear()
        print(f"california {written} sections", flush=True)

    for section in chosen.values():
        code_name = codes.get(section["code"], section["code"])
        number = section["number"].rstrip(".")
        path = [
            *section_path(code_name, toc.get(section["code"], []), section),
            f"Section {number}",
        ]
        text = lob_to_text(archive.read(section["lob"]).decode("utf-8", "replace"))
        doc = {
            "state": "CA",
            "code": section["code"],
            "citation": f"{section['code']} § {number}",
            "title": f"Section {number}",
            "url": section_url(section["code"], section["number"]),
            "text": text,
            "section": path,
        }
        batch.append(
            UpdateOne(
                {"url": doc["url"]},
                {"$set": doc, "$setOnInsert": {"loaded_at": loaded_at}},
                upsert=True,
            )
        )
        if len(batch) >= 500:
            flush()
    flush()
    print(f"california done {written}", flush=True)
    archive.close()
    return {"sections": written}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args()
    load(args.zip)


if __name__ == "__main__":
    main()
