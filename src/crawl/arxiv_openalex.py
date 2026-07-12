"""Crawl earthquake research papers.

Two complementary sources, both free and key-less:

  * OpenAlex  -- broad metadata + abstracts for works with "earthquake" in the
                 title. Abstracts are stored as an inverted index; we rebuild
                 plain text from it. This gives us many short, clean documents.

  * arXiv     -- open full-text PDFs. We query the arXiv API for papers with
                 "earthquake" in the title, download the PDF, and extract text
                 with PyMuPDF. Fewer docs but much longer.

Usage:
  python -m src.crawl.arxiv_openalex openalex --max 2000
  python -m src.crawl.arxiv_openalex arxiv    --max 300
"""
from __future__ import annotations

import argparse
import io
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import DATA_RAW, Doc, http_get, write_jsonl

# ----------------------------------------------------------------------------
# OpenAlex
# ----------------------------------------------------------------------------
OPENALEX = "https://api.openalex.org/works"

# High-signal seismology / general-science venues. Resolved to OpenAlex source
# IDs (see `python -m src.crawl.arxiv_openalex journals`). Broad journals like
# Nature/Science rarely put "earthquake" in the title, so the journal crawl
# searches title+abstract to actually capture their earthquake papers.
JOURNALS: dict[str, str] = {
    "Science": "S3880285",
    "Nature": "S137773608",
    "Nature Geoscience": "S48977010",
    "Nature Communications": "S64187185",
    "Geophysical Research Letters": "S36624081",
    "Journal of Geophysical Research: Solid Earth": "S4210228715",
    "Earth and Planetary Science Letters": "S119230507",
    "Seismological Research Letters": "S183957208",
    "Geophysical Journal International": "S108821158",
    "The Seismic Record": "S4210235748",
    "Seismica": "S4387284412",
}


def _venue_of(work: dict) -> str:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name", "") or ""


def _work_to_doc(w: dict) -> Doc | None:
    """Convert an OpenAlex work record to a Doc, or None if no abstract."""
    abstract = _abstract_from_inverted_index(w.get("abstract_inverted_index"))
    if not abstract:
        return None
    title = w.get("title") or ""
    return Doc(
        source="openalex",
        id=w.get("id", "").rsplit("/", 1)[-1],
        title=title,
        text=f"{title}\n\n{abstract}",
        url=w.get("id", ""),
        date=w.get("publication_date", ""),
        extra={
            "cited_by": w.get("cited_by_count", 0),
            "venue": _venue_of(w),
        },
    )


def iter_works_raw(filter_str: str, max_works: int | None = None):
    """Yield raw OpenAlex work dicts for a `filter=` query, cursor-paginated.

    max_works=None paginates until the result set is exhausted. Lazy: callers
    can break early (e.g. once they have enough usable full-texts).
    """
    cursor = "*"
    n = 0
    while max_works is None or n < max_works:
        params = {
            "filter": filter_str,
            "per-page": 200,
            "cursor": cursor,
            "mailto": "jiazhe868@gmail.com",
        }
        raw = http_get(OPENALEX, params=params, min_interval=1.0)
        if raw is None:
            break
        data = json.loads(raw)
        results = data.get("results", [])
        if not results:
            break
        for w in results:
            yield w
            n += 1
            if max_works is not None and n >= max_works:
                break
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break


def _iter_openalex(filter_str: str, max_docs: int):
    """Yield abstract-based Doc objects for an OpenAlex `filter=` query."""
    seen = 0
    for w in iter_works_raw(filter_str):
        doc = _work_to_doc(w)
        if doc is None:
            continue
        yield doc
        seen += 1
        if seen >= max_docs:
            break


def _abstract_from_inverted_index(inv: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; reconstruct text."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def crawl_openalex(max_docs: int, out: Path) -> int:
    """Broad crawl: any work with 'earthquake' in the title."""
    docs: list[Doc] = []
    for doc in _iter_openalex("title.search:earthquake", max_docs):
        docs.append(doc)
        if len(docs) % 500 == 0:
            print(f"[openalex] collected {len(docs)}")
    n = write_jsonl(docs, out)
    print(f"[openalex] wrote {n} docs -> {out}")
    return n


def crawl_journals(max_per_journal: int, out: Path, title_only: bool) -> int:
    """Targeted crawl of the curated JOURNALS list for earthquake papers.

    By default searches title+abstract (broad journals rarely title papers
    "earthquake"); pass title_only=True to restrict to the title.
    """
    field = "title.search" if title_only else "title_and_abstract.search"
    seen_ids: set[str] = set()
    docs: list[Doc] = []
    for name, sid in JOURNALS.items():
        flt = f"primary_location.source.id:{sid},{field}:earthquake"
        got = 0
        for doc in _iter_openalex(flt, max_per_journal):
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            docs.append(doc)
            got += 1
        print(f"[journals] {name:44s} +{got}")
    n = write_jsonl(docs, out)
    print(f"[journals] wrote {n} docs from {len(JOURNALS)} venues -> {out}")
    return n


# ----------------------------------------------------------------------------
# arXiv
# ----------------------------------------------------------------------------
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    import fitz  # PyMuPDF, imported lazily so metadata-only runs don't need it

    text_parts = []
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def crawl_arxiv(max_docs: int, out: Path) -> int:
    docs: list[Doc] = []
    start = 0
    batch = 50
    while len(docs) < max_docs:
        params = {
            "search_query": 'ti:"earthquake"',
            "start": start,
            "max_results": batch,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        raw = http_get(ARXIV_API, params=params, min_interval=3.0)  # arXiv: slow
        if raw is None:
            break
        root = ET.fromstring(raw)
        entries = root.findall(f"{ATOM}entry")
        if not entries:
            break
        for e in entries:
            arxiv_id = e.findtext(f"{ATOM}id", "").rsplit("/", 1)[-1]
            title = " ".join((e.findtext(f"{ATOM}title") or "").split())
            pdf_url = ""
            for link in e.findall(f"{ATOM}link"):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
            if not pdf_url:
                continue
            pdf = http_get(pdf_url, min_interval=3.0)
            if pdf is None:
                continue
            try:
                text = _extract_pdf_text(pdf)
            except Exception as ex:  # noqa: BLE001 - PDF parsing is flaky
                print(f"[arxiv] pdf parse failed {arxiv_id}: {ex}")
                continue
            docs.append(
                Doc(
                    source="arxiv",
                    id=arxiv_id,
                    title=title,
                    text=text,
                    url=pdf_url,
                    date=e.findtext(f"{ATOM}published", "")[:10],
                )
            )
            print(f"[arxiv] {len(docs)}: {title[:70]}")
            if len(docs) >= max_docs:
                break
        start += batch
        time.sleep(1.0)
    n = write_jsonl(docs, out)
    print(f"[arxiv] wrote {n} docs -> {out}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["openalex", "arxiv", "journals"])
    parser.add_argument(
        "--max",
        type=int,
        default=1000,
        help="max docs (for 'journals': max per journal)",
    )
    parser.add_argument(
        "--title-only",
        action="store_true",
        help="journals: match 'earthquake' in title only (default: title+abstract)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.source == "openalex":
        crawl_openalex(args.max, args.out or DATA_RAW / "openalex.jsonl")
    elif args.source == "journals":
        crawl_journals(
            args.max, args.out or DATA_RAW / "journals.jsonl", args.title_only
        )
    else:
        crawl_arxiv(args.max, args.out or DATA_RAW / "arxiv.jsonl")


if __name__ == "__main__":
    main()
