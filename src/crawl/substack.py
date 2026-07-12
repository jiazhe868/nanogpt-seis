"""Crawl the 'Earthquake Insights' Substack (Judith Hubbard & Kyle Bradley).

Substack exposes a sitemap and a JSON archive API. We:
  1. Page through the archive API to list post slugs/urls.
  2. Fetch each post HTML and extract the article body text.

Only public / free-preview content is retrieved. Paywalled bodies simply come
back short and get filtered downstream.

Usage:
  python -m src.crawl.substack --max 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup

from .common import DATA_RAW, Doc, http_get, write_jsonl

BASE = "https://earthquakeinsights.substack.com"


def list_posts(max_posts: int) -> list[dict]:
    """Use Substack's archive API: /api/v1/archive?sort=new&offset=&limit=."""
    posts: list[dict] = []
    offset = 0
    limit = 50
    while len(posts) < max_posts:
        url = f"{BASE}/api/v1/archive"
        raw = http_get(
            url,
            params={"sort": "new", "offset": offset, "limit": limit},
            min_interval=1.0,
        )
        if raw is None:
            break
        batch = json.loads(raw)
        if not batch:
            break
        posts.extend(batch)
        offset += limit
        print(f"[substack] listed {len(posts)}")
    return posts[:max_posts]


def extract_body(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Substack wraps the article body in div.available-content / div.body.
    body = soup.select_one("div.available-content") or soup.select_one(
        "div.body.markup"
    )
    if body is None:
        return ""
    for tag in body.select("script, style, figure, .subscription-widget-wrap"):
        tag.decompose()
    return body.get_text("\n", strip=True)


def crawl(max_posts: int, out: Path) -> int:
    posts = list_posts(max_posts)
    print(f"[substack] {len(posts)} posts to fetch")
    docs: list[Doc] = []
    for p in posts:
        url = p.get("canonical_url") or f"{BASE}/p/{p.get('slug', '')}"
        html = http_get(url, min_interval=1.0)
        if html is None:
            continue
        text = extract_body(html)
        # Prepend subtitle if present for a little more signal.
        subtitle = p.get("subtitle") or ""
        title = p.get("title") or ""
        full = "\n\n".join(x for x in [title, subtitle, text] if x)
        docs.append(
            Doc(
                source="substack",
                id=f"substack-{p.get('id', p.get('slug', ''))}",
                title=title,
                text=full,
                url=url,
                date=(p.get("post_date") or "")[:10],
            )
        )
    n = write_jsonl(docs, out)
    print(f"[substack] wrote {n} docs -> {out}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=500)
    parser.add_argument("--out", type=Path, default=DATA_RAW / "substack.jsonl")
    args = parser.parse_args()
    crawl(args.max, args.out)


if __name__ == "__main__":
    main()
