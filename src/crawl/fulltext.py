"""Crawl open-access FULL-TEXT PDFs for earthquake papers.

Enumeration backend (default `crossref`, since OpenAlex now charges per request):
  * Crossref   -- list earthquake journal-article DOIs (free, deep cursor
                  paging, filterable by journal ISSN). Gives title / venue /
                  date / (sometimes) a JATS abstract.
  * Unpaywall  -- resolve each DOI to its open-access PDF locations (free,
                  100k/day). Repository (green) locations are tried first; they
                  are far more reliably downloadable than publisher links.

Robustness (unchanged from before):
  * Every download is validated: %PDF magic, >= 2 pages, >= --min-fulltext-chars
    of extracted text. Otherwise the next candidate is tried, then the abstract.
  * Concurrent: a thread pool downloads/extracts many PDFs at once; a per-host
    throttle keeps each server politely spaced while different hosts run in
    parallel. Crossref pagination + file writes stay on the main thread.
  * Resumable: output is appended and already-seen ids are skipped.

Usage:
  python -m src.crawl.fulltext --broad 30000 --per-journal 3000 --workers 64
  python -m src.crawl.fulltext --backend openalex ...        # legacy (paid now)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import io
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .arxiv_openalex import (
    JOURNALS,
    _abstract_from_inverted_index,
    _venue_of,
    iter_works_raw,
)
from .common import DATA_RAW, USER_AGENT, Doc

EMAIL = "jiazhe868@gmail.com"
CROSSREF = "https://api.crossref.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/"

# Curated venues -> ISSN (for Crossref filtering).
JOURNAL_ISSNS: dict[str, str] = {
    "Science": "0036-8075",
    "Nature": "0028-0836",
    "Nature Geoscience": "1752-0894",
    "Nature Communications": "2041-1723",
    "Geophysical Research Letters": "0094-8276",
    "Journal of Geophysical Research: Solid Earth": "2169-9313",
    "Earth and Planetary Science Letters": "0012-821X",
    "Seismological Research Letters": "0895-0695",
    "Geophysical Journal International": "0956-540X",
    "The Seismic Record": "2694-4006",
    "Seismica": "2816-9387",
}

# Shared, thread-safe session with a large connection pool (many hosts + a
# high-volume single host, Unpaywall).
_SESSION = requests.Session()
_ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=64, pool_maxsize=128, max_retries=0
)
_SESSION.mount("http://", _ADAPTER)
_SESSION.mount("https://", _ADAPTER)


class HostThrottle:
    """Per-host minimum spacing between request *starts*.

    Different hosts run concurrently (independent locks); repeated hits to the
    same host are spaced by `min_interval` so we stay polite under parallelism.
    """

    def __init__(self, min_interval: float):
        self.min = min_interval
        self._locks: dict[str, threading.Lock] = {}
        self._last: dict[str, float] = {}
        self._guard = threading.Lock()

    def wait(self, host: str) -> None:
        with self._guard:
            host_lock = self._locks.setdefault(host, threading.Lock())
            self._last.setdefault(host, 0.0)
        with host_lock:
            delta = self.min - (time.monotonic() - self._last[host])
            if delta > 0:
                time.sleep(delta)
            self._last[host] = time.monotonic()


# --------------------------------------------------------------------------- #
# PDF download + extraction (shared by both backends)
# --------------------------------------------------------------------------- #
def _extract_fulltext(
    url: str, throttle: HostThrottle, min_chars: int, max_mb: int
) -> str | None:
    """Download `url` (host-throttled), verify a real PDF, return text or None."""
    import fitz  # PyMuPDF

    throttle.wait(urlparse(url).netloc)
    try:
        resp = _SESSION.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        if resp.status_code != 200:
            return None
        pdf_bytes = resp.content
    except requests.RequestException:
        return None
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        return None                       # missing / HTML landing / stub
    if len(pdf_bytes) > max_mb * 1024 * 1024:
        return None
    try:
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            if doc.page_count < 2:
                return None
            text = "\n".join(page.get_text() for page in doc)
    except Exception:                     # noqa: BLE001 - PDF parsing is flaky
        return None
    if len(text) < min_chars:
        return None
    return text


def _load_done_ids(out: Path) -> set[str]:
    done: set[str] = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    continue
    return done


# --------------------------------------------------------------------------- #
# Crossref + Unpaywall backend
# --------------------------------------------------------------------------- #
def iter_crossref(query: str | None, issn: str | None):
    """Yield Crossref journal-article records, deep cursor-paginated."""
    from .common import http_get

    cursor = "*"
    while True:
        filters = ["type:journal-article"]
        if issn:
            filters.append(f"issn:{issn}")
        params = {
            "rows": 1000,
            "cursor": cursor,
            "filter": ",".join(filters),
            "select": "DOI,title,container-title,issued,abstract",
            "mailto": EMAIL,
        }
        if query:
            params["query.bibliographic"] = query
        raw = http_get(CROSSREF, params=params, min_interval=0.5, use_cache=False)
        if raw is None:
            return
        msg = json.loads(raw).get("message", {})
        items = msg.get("items", [])
        if not items:
            return
        for it in items:
            yield it
        cursor = msg.get("next-cursor")
        if not cursor:
            return


def _unpaywall_pdf_urls(doi: str, api_throttle: HostThrottle) -> list[str]:
    """Prioritized OA PDF URLs for a DOI (repository/green first)."""
    api_throttle.wait("api.unpaywall.org")
    try:
        r = _SESSION.get(
            UNPAYWALL + doi,
            params={"email": EMAIL},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if r.status_code != 200:
            return []
        j = r.json()
    except (requests.RequestException, ValueError):
        return []
    scored: list[tuple[int, str]] = []
    for loc in j.get("oa_locations") or []:
        url = loc.get("url_for_pdf")
        if not url:
            continue
        prio = 0 if loc.get("host_type") == "repository" else 1
        scored.append((prio, url))
    seen: set[str] = set()
    ordered: list[str] = []
    for _, url in sorted(scored, key=lambda x: x[0]):
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _crossref_abstract(item: dict) -> str:
    abstract = item.get("abstract")
    if not abstract:
        return ""
    abstract = re.sub(r"<[^>]+>", " ", abstract)      # strip JATS tags
    return " ".join(html.unescape(abstract).split())


def _crossref_date(item: dict) -> str:
    date_parts = (item.get("issued") or {}).get("date-parts") or [[]]
    parts = date_parts[0] if date_parts else []
    return "-".join(str(p) for p in parts) if parts else ""


def _crossref_to_doc(
    item: dict,
    api_throttle: HostThrottle,
    pdf_throttle: HostThrottle,
    min_chars: int,
    max_mb: int,
    keep_abstract: bool,
) -> Doc | None:
    doi = (item.get("DOI") or "").lower()
    if not doi:
        return None
    titles = item.get("title") or [""]
    title = " ".join((titles[0] or "").split())
    venues = item.get("container-title") or [""]
    extra = {"venue": venues[0] if venues else "", "doi": doi}
    date = _crossref_date(item)

    for url in _unpaywall_pdf_urls(doi, api_throttle):
        text = _extract_fulltext(url, pdf_throttle, min_chars, max_mb)
        if text:
            return Doc(source="fulltext", id=doi, title=title,
                       text=f"{title}\n\n{text}", url=url, date=date,
                       extra={**extra, "full_text": True})
    if keep_abstract:
        abstract = _crossref_abstract(item)
        if abstract and len(abstract) >= 200:
            return Doc(source="fulltext", id=doi, title=title,
                       text=f"{title}\n\n{abstract}", url=f"https://doi.org/{doi}",
                       date=date, extra={**extra, "full_text": False})
    return None


# --------------------------------------------------------------------------- #
# Legacy OpenAlex backend (kept for reference; OpenAlex now charges per request)
# --------------------------------------------------------------------------- #
def _candidate_pdf_urls(work: dict) -> list[str]:
    scored: list[tuple[int, str]] = []
    for loc in work.get("locations") or []:
        if not loc.get("is_oa"):
            continue
        url = loc.get("pdf_url")
        if not url:
            continue
        src = loc.get("source") or {}
        scored.append((0 if src.get("type") == "repository" else 1, url))
    bol = work.get("best_oa_location") or {}
    if bol.get("pdf_url"):
        scored.append((2, bol["pdf_url"]))
    seen: set[str] = set()
    ordered: list[str] = []
    for _, url in sorted(scored, key=lambda x: x[0]):
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _work_to_fulltext_doc(work, throttle, min_chars, max_mb, keep_abstract):
    wid = work.get("id", "").rsplit("/", 1)[-1]
    title = work.get("title") or ""
    extra = {"cited_by": work.get("cited_by_count", 0), "venue": _venue_of(work),
             "oa_status": (work.get("open_access") or {}).get("oa_status", "")}
    for url in _candidate_pdf_urls(work):
        text = _extract_fulltext(url, throttle, min_chars, max_mb)
        if text:
            return Doc(source="fulltext", id=wid, title=title,
                       text=f"{title}\n\n{text}", url=url,
                       date=work.get("publication_date", ""),
                       extra={**extra, "full_text": True})
    if keep_abstract:
        abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
        if abstract:
            return Doc(source="fulltext", id=wid, title=title,
                       text=f"{title}\n\n{abstract}", url=work.get("id", ""),
                       date=work.get("publication_date", ""),
                       extra={**extra, "full_text": False})
    return None


# --------------------------------------------------------------------------- #
# Concurrent driver (backend-agnostic)
# --------------------------------------------------------------------------- #
def _crawl_target(name, item_iter, id_of, process, cap, min_hit,
                  executor, fout, done, args):
    got_ft = got_ab = scanned = 0
    max_scan = cap * args.scan_multiple + 50
    window = args.workers * 3
    pending: dict[cf.Future, str] = {}

    def refill() -> None:
        nonlocal scanned
        while len(pending) < window and scanned < max_scan and got_ft < cap:
            try:
                item = next(item_iter)
            except StopIteration:
                return
            scanned += 1
            iid = id_of(item)
            if not iid or iid in done:
                continue
            pending[executor.submit(process, item)] = iid

    refill()
    while pending:
        finished, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
        for fut in finished:
            pending.pop(fut, None)
            try:
                doc = fut.result()
            except Exception:  # noqa: BLE001
                doc = None
            if doc is None or doc.id in done:
                continue
            fout.write(doc.to_json() + "\n")
            fout.flush()
            done.add(doc.id)
            if doc.extra.get("full_text"):
                got_ft += 1
            else:
                got_ab += 1
            if (got_ft + got_ab) % 25 == 0:
                print(f"[fulltext] {name}: {got_ft} full / {got_ab} abstr "
                      f"({scanned} scanned)", end="\r", flush=True)
        # Gate on `scanned`: each scanned paper costs one Unpaywall call, so this
        # bounds wasted API budget on low-yield (paywalled) journals to ~abort_after.
        low_yield = (min_hit > 0 and scanned >= args.abort_after
                     and got_ft / max(1, scanned) < min_hit)
        if got_ft >= cap or low_yield:
            for f in pending:
                f.cancel()
            if low_yield:
                print(f"[fulltext] {name}: low yield "
                      f"({got_ft}/{scanned} scanned), skipping.        ")
            break
        refill()
    return got_ft, got_ab, scanned


def crawl(args: argparse.Namespace) -> None:
    out: Path = args.out
    done = _load_done_ids(out)
    print(f"[fulltext] backend={args.backend}  resuming: {len(done)} ids in {out.name}")

    pdf_throttle = HostThrottle(args.host_interval)
    api_throttle = HostThrottle(args.api_interval)
    keep_abs = not args.no_abstract_fallback
    mc, mm = args.min_fulltext_chars, args.max_mb

    targets: list[tuple] = []
    if args.backend == "crossref":
        id_of = lambda it: (it.get("DOI") or "").lower()  # noqa: E731
        proc = lambda it: _crossref_to_doc(  # noqa: E731
            it, api_throttle, pdf_throttle, mc, mm, keep_abs)
        # Journals first: their OA hit-rate per Unpaywall call is far higher than
        # the broad pool (~2-3%), so the daily Unpaywall budget is best spent
        # here. Broad runs last with whatever budget remains.
        for name, issn in JOURNAL_ISSNS.items():
            targets.append((name, iter_crossref("earthquake", issn),
                            id_of, proc, args.per_journal, args.min_hit))
        if not args.no_broad:
            targets.append(("broad", iter_crossref("earthquake", None),
                            id_of, proc, args.broad, 0.0))  # broad: no yield gate
    else:  # openalex (legacy)
        field = "title.search" if args.title_only else "title_and_abstract.search"
        id_of = lambda w: w.get("id", "").rsplit("/", 1)[-1]  # noqa: E731
        proc = lambda w: _work_to_fulltext_doc(  # noqa: E731
            w, pdf_throttle, mc, mm, keep_abs)
        if not args.no_broad:
            targets.append(("broad",
                            iter_works_raw(f"{field}:earthquake,is_oa:true"),
                            id_of, proc, args.broad, 0.0))
        for name, sid in JOURNALS.items():
            flt = f"primary_location.source.id:{sid},{field}:earthquake,is_oa:true"
            targets.append((name, iter_works_raw(flt), id_of, proc,
                            args.per_journal, args.min_hit))

    out.parent.mkdir(parents=True, exist_ok=True)
    fout = out.open("a", encoding="utf-8")
    total_full = total_abstract = 0
    t0 = time.monotonic()
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for name, it, id_of, proc, cap, min_hit in targets:
                if cap <= 0:
                    continue
                n_full, n_abstract, n_scanned = _crawl_target(
                    name, it, id_of, proc, cap, min_hit, pool, fout, done, args)
                total_full += n_full
                total_abstract += n_abstract
                rate = total_full / max(1e-6, (time.monotonic() - t0) / 60)
                print(f"[fulltext] {name:44s} full={n_full:5d} abstr={n_abstract:5d} "
                      f"scanned={n_scanned}  ({rate:.0f} full/min, {total_full} total)")
    finally:
        fout.close()
    print(f"[fulltext] DONE  full_text={total_full}  abstract_fallback={total_abstract}"
          f"  -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["crossref", "openalex"], default="crossref")
    parser.add_argument("--per-journal", type=int, default=3000)
    parser.add_argument("--broad", type=int, default=30000)
    parser.add_argument("--no-broad", action="store_true")
    parser.add_argument("--title-only", action="store_true",
                    help="(openalex only) match title instead of title+abstract")
    parser.add_argument("--min-fulltext-chars", type=int, default=4000)
    parser.add_argument("--max-mb", type=int, default=60)
    parser.add_argument("--scan-multiple", type=int, default=6)
    parser.add_argument("--abort-after", type=int, default=400,
                    help="processed papers before applying the low-yield gate")
    parser.add_argument("--min-hit", type=float, default=0.04,
                    help="skip a journal if full-text hit-rate falls below this")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--host-interval", type=float, default=1.0,
                    help="min seconds between requests to the same PDF host")
    parser.add_argument("--api-interval", type=float, default=0.05,
                    help="min seconds between Unpaywall API calls")
    parser.add_argument("--no-abstract-fallback", action="store_true")
    parser.add_argument("--out", type=Path, default=DATA_RAW / "fulltext.jsonl")
    args = parser.parse_args()
    crawl(args)


if __name__ == "__main__":
    main()
