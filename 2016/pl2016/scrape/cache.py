"""Raw HTML cache. The crawler writes a page here before parsing it."""

from __future__ import annotations

import hashlib
from pathlib import Path


def html_path(root: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return root / "html" / f"{digest}.html"


def write_html(root: Path, url: str, body: bytes) -> Path:
    path = html_path(root, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def read_html(root: Path, url: str) -> bytes | None:
    path = html_path(root, url)
    if not path.exists():
        return None
    return path.read_bytes()
