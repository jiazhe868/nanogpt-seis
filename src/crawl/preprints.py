"""Crawl free full-text preprints: arXiv + EarthArXiv (OSF).

Unlike the Crossref+Unpaywall path, preprint servers host the full text openly,
so the hit-rate is ~100% and there's no per-DOI OA lottery / API budget. This is
the most reliable free source of full-text seismology papers.

  * arXiv     -- export API, `all:earthquake`; PDF from arxiv.org.
  * EarthArXiv -- OSF preprints API (provider=eartharxiv), client-side filtered
                  to seismology; PDF from osf.io/<guid>/download.

Writes to a SEPARATE file (data/raw/preprints.jsonl) so it can run concurrently
with the Crossref crawl without interleaving appends. Resumable.

Usage:
  python -m src.crawl.preprints --arxiv 3000 --eartharxiv 1000 --workers 16
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import types
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import DATA_RAW, Doc, http_get
from .fulltext import (
    HostThrottle,
    _crawl_target,
    _extract_fulltext,
    _load_done_ids,
    _unpaywall_pdf_urls,
)

ATOM = "{http://www.w3.org/2005/Atom}"
SEISMO_KW = ("earthquake", "seismic", "seismol", "fault", "rupture", "tremor",
             "tectonic", "subduction", "aftershock", "ground motion")


# --------------------------------------------------------------------------- #
def iter_arxiv(query: str, page_size: int = 100):
    start = 0
    while True:
        raw = http_get(
            "http://export.arxiv.org/api/query",
            params={"search_query": query, "start": start,
                    "max_results": page_size, "sortBy": "submittedDate",
                    "sortOrder": "descending"},
            min_interval=3.0, use_cache=False,        # arXiv asks for ~3s spacing
        )
        if raw is None:
            return
        root = ET.fromstring(raw)
        entries = root.findall(f"{ATOM}entry")
        if not entries:
            return
        for e in entries:
            aid = (e.findtext(f"{ATOM}id") or "").rsplit("/", 1)[-1]
            title = " ".join((e.findtext(f"{ATOM}title") or "").split())
            summary = " ".join((e.findtext(f"{ATOM}summary") or "").split())
            pdf = None
            for link in e.findall(f"{ATOM}link"):
                if link.get("title") == "pdf":
                    pdf = link.get("href")
            yield {"id": f"arxiv-{aid}", "title": title, "summary": summary,
                   "pdf": pdf or f"https://arxiv.org/pdf/{aid}",
                   "date": (e.findtext(f"{ATOM}published") or "")[:10],
                   "venue": "arXiv"}
        start += page_size


def iter_eartharxiv():
    url = ("https://api.osf.io/v2/preprints/?filter[provider]=eartharxiv"
           "&page[size]=100")
    while url:
        raw = http_get(url, min_interval=1.0, use_cache=False)
        if raw is None:
            return
        j = json.loads(raw)
        for p in j.get("data", []):
            a = p.get("attributes", {})
            title = a.get("title") or ""
            desc = a.get("description") or ""
            if not any(k in (title + " " + desc).lower() for k in SEISMO_KW):
                continue
            guid = p["id"]
            # Versioned OSF preprints don't expose a working /download URL, but
            # their DOI resolves to a real PDF via Unpaywall (host eartharxiv.org).
            doi_link = (p.get("links") or {}).get("preprint_doi", "")
            doi = doi_link.replace("https://doi.org/", "").lower()
            if not doi:
                continue
            yield {"id": f"eartharxiv-{guid}", "title": title, "summary": desc,
                   "doi": doi, "date": (a.get("date_published") or "")[:10],
                   "venue": "EarthArXiv"}
        url = (j.get("links") or {}).get("next")


def _preprint_doc(item, throttle, api_throttle, min_chars, max_mb, keep_abstract):
    src = "arxiv" if item["id"].startswith("arxiv-") else "eartharxiv"
    # arXiv has a direct PDF; EarthArXiv is resolved via its DOI through Unpaywall.
    urls = [item["pdf"]] if "pdf" in item else _unpaywall_pdf_urls(item["doi"], api_throttle)
    for url in urls:
        text = _extract_fulltext(url, throttle, min_chars, max_mb)
        if text:
            return Doc(source=src, id=item["id"], title=item["title"],
                       text=f"{item['title']}\n\n{text}", url=url,
                       date=item["date"],
                       extra={"venue": item["venue"], "full_text": True})
    if keep_abstract and len(item.get("summary", "")) >= 200:
        return Doc(source=src, id=item["id"], title=item["title"],
                   text=f"{item['title']}\n\n{item['summary']}", url=item["pdf"],
                   date=item["date"],
                   extra={"venue": item["venue"], "full_text": False})
    return None


def crawl(args: argparse.Namespace) -> None:
    out: Path = args.out
    done = _load_done_ids(out)
    print(f"[preprints] resuming: {len(done)} ids in {out.name}")
    throttle = HostThrottle(args.host_interval)
    api_throttle = HostThrottle(0.1)          # Unpaywall (for EarthArXiv DOIs)
    keep_abs = not args.no_abstract_fallback
    proc = lambda it: _preprint_doc(  # noqa: E731
        it, throttle, api_throttle, args.min_fulltext_chars, args.max_mb, keep_abs)
    idf = lambda it: it["id"]  # noqa: E731
    # No yield gate for preprints (~100% full text) -> scan_multiple high, min_hit 0.
    fake = types.SimpleNamespace(workers=args.workers, scan_multiple=1000,
                                 abort_after=10 ** 9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fout = out.open("a", encoding="utf-8")
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for name, it, cap in [("arxiv", iter_arxiv("all:earthquake"), args.arxiv),
                                  ("eartharxiv", iter_eartharxiv(), args.eartharxiv)]:
                if cap <= 0:
                    continue
                ft, ab, sc = _crawl_target(name, it, idf, proc, cap, 0.0,
                                           ex, fout, done, fake)
                print(f"[preprints] {name:12s} full={ft} abstr={ab} scanned={sc}")
    finally:
        fout.close()
    print(f"[preprints] DONE -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv", type=int, default=3000)
    parser.add_argument("--eartharxiv", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--host-interval", type=float, default=1.0)
    parser.add_argument("--min-fulltext-chars", type=int, default=4000)
    parser.add_argument("--max-mb", type=int, default=60)
    parser.add_argument("--no-abstract-fallback", action="store_true")
    parser.add_argument("--out", type=Path, default=DATA_RAW / "preprints.jsonl")
    args = parser.parse_args()
    crawl(args)


if __name__ == "__main__":
    main()
