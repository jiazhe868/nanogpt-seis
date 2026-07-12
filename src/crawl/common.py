"""Shared helpers for the crawlers: polite HTTP, on-disk cache, JSONL sink."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import requests

# Be a good citizen: identify ourselves and go slow.
USER_AGENT = (
    "nanogpt-seis-research-crawler/0.1 "
    "(educational LLM pretraining; contact: jiazhe868@gmail.com)"
)

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
CACHE_DIR = DATA_RAW / ".httpcache"


@dataclass
class Doc:
    """One training document. `text` is the cleaned body we will tokenize."""

    source: str          # "arxiv" | "openalex" | "wikipedia" | "substack"
    id: str              # stable per-source id (used for dedup)
    title: str
    text: str
    url: str = ""
    date: str = ""       # ISO-ish, best effort
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _cache_path(key: str) -> Path:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / h[:2] / f"{h}.bin"


def http_get(
    url: str,
    *,
    params: Optional[dict] = None,
    min_interval: float = 1.0,
    timeout: float = 30.0,
    use_cache: bool = True,
    max_retries: int = 4,
) -> Optional[bytes]:
    """GET with on-disk cache, retries, and a per-call minimum delay.

    Returns raw bytes, or None if the request ultimately failed.
    """
    cache_key = url + ("?" + json.dumps(params, sort_keys=True) if params else "")
    cpath = _cache_path(cache_key)
    if use_cache and cpath.exists():
        return cpath.read_bytes()

    headers = {"User-Agent": USER_AGENT}
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=timeout
            )
            if resp.status_code == 200:
                if use_cache:
                    cpath.parent.mkdir(parents=True, exist_ok=True)
                    cpath.write_bytes(resp.content)
                time.sleep(min_interval)
                return resp.content
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue
            # 4xx other than 429: not worth retrying.
            print(f"[http] {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            print(f"[http] error {e} (attempt {attempt + 1})")
            time.sleep(backoff)
            backoff *= 2
    return None


def write_jsonl(docs: Iterable[Doc], path: Path) -> int:
    """Append docs to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            if d.text and d.text.strip():
                f.write(d.to_json() + "\n")
                n += 1
    return n
