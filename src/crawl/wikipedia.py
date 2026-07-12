"""Crawl Wikipedia articles whose title contains 'earthquake'.

Strategy:
  1. Use the MediaWiki search API (srsearch=intitle:earthquake) to enumerate
     page titles.
  2. For each page, pull the plain-text extract via the extracts API.

Usage:
  python -m src.crawl.wikipedia --max-pages 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DATA_RAW, Doc, http_get, write_jsonl

API = "https://en.wikipedia.org/w/api.php"


def search_titles(query: str, max_pages: int) -> list[dict]:
    """Enumerate pages matching `query` via the search API (paginated)."""
    results: list[dict] = []
    sroffset = 0
    while len(results) < max_pages:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 50,
            "sroffset": sroffset,
            "srnamespace": 0,      # main article namespace only
            "format": "json",
        }
        raw = http_get(API, params=params, min_interval=0.5)
        if raw is None:
            break
        data = json.loads(raw)
        hits = data.get("query", {}).get("search", [])
        if not hits:
            break
        results.extend(hits)
        cont = data.get("continue", {})
        if "sroffset" not in cont:
            break
        sroffset = cont["sroffset"]
    return results[:max_pages]


def fetch_extract(pageid: int) -> tuple[str, str, str]:
    """Return (title, plaintext, url) for one page id."""
    params = {
        "action": "query",
        "pageids": pageid,
        "prop": "extracts|info",
        "explaintext": 1,          # strip wiki markup / HTML
        "exsectionformat": "plain",
        "inprop": "url",
        "format": "json",
    }
    raw = http_get(API, params=params, min_interval=0.3)
    if raw is None:
        return "", "", ""
    page = json.loads(raw)["query"]["pages"][str(pageid)]
    return page.get("title", ""), page.get("extract", ""), page.get("fullurl", "")


def crawl(max_pages: int, out: Path) -> int:
    hits = search_titles("intitle:earthquake", max_pages)
    print(f"[wikipedia] {len(hits)} candidate pages")
    docs: list[Doc] = []
    for h in hits:
        title, text, url = fetch_extract(h["pageid"])
        if not text:
            continue
        docs.append(
            Doc(
                source="wikipedia",
                id=f"wiki-{h['pageid']}",
                title=title,
                text=text,
                url=url,
            )
        )
    n = write_jsonl(docs, out)
    print(f"[wikipedia] wrote {n} docs -> {out}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--out", type=Path, default=DATA_RAW / "wikipedia.jsonl")
    args = parser.parse_args()
    crawl(args.max_pages, args.out)


if __name__ == "__main__":
    main()
