"""Stream curated general / encyclopedic text to balance the earthquake corpus.

The pretraining corpus is ~90% research-paper prose, which is narrow in register
and hurts the base model's plain-language fluency. This adds two high-quality
general sources, streamed (no full download) up to a token budget, in the same
`Doc` JSONL schema so build_corpus picks them up automatically:

  * Wikipedia (English)  — encyclopedic backbone (clean, factual).
  * FineWeb-Edu          — web text filtered for educational quality (fluent,
                           coherent prose — the exact weakness of a paper-only mix).

Usage:
  python -m src.crawl.general --wiki-tokens 120_000_000 --fineweb-tokens 120_000_000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import DATA_RAW, Doc


def _stream_to_budget(examples, *, source, budget, min_chars,
                      id_of, text_of, title_of, url_of, toks_of, out: Path) -> int:
    """Write docs from a streaming dataset until `budget` (estimated) tokens."""
    out.parent.mkdir(parents=True, exist_ok=True)
    written = toks = 0
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            text = text_of(ex)
            if not text or len(text) < min_chars:
                continue
            doc = Doc(source=source, id=id_of(ex), title=title_of(ex),
                      text=text, url=url_of(ex))
            f.write(doc.to_json() + "\n")
            written += 1
            toks += toks_of(ex, text)
            if written % 5000 == 0:
                print(f"[general] {source}: {written} docs, "
                      f"{toks/1e6:.1f}M tokens", flush=True)
            if toks >= budget:
                break
    print(f"[general] {source}: DONE {written} docs, ~{toks/1e6:.1f}M tokens -> {out}")
    return written


def crawl(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    if args.wiki_tokens > 0:
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
        _stream_to_budget(
            wiki, source="wikipedia", budget=args.wiki_tokens,
            min_chars=args.min_chars,
            id_of=lambda e: f"wiki-{e['id']}",
            text_of=lambda e: e["text"],
            title_of=lambda e: e.get("title", ""),
            url_of=lambda e: e.get("url", ""),
            toks_of=lambda e, t: len(t) // 4,          # ~4 chars/token estimate
            out=args.out_dir / "wiki_general.jsonl",
        )

    if args.fineweb_tokens > 0:
        fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                          split="train", streaming=True)
        if args.min_score > 0:
            fw = fw.filter(lambda e: (e.get("score") or 0) >= args.min_score)
        _stream_to_budget(
            fw, source="fineweb_edu", budget=args.fineweb_tokens,
            min_chars=args.min_chars,
            id_of=lambda e: f"fineweb-{e['id']}",
            text_of=lambda e: e["text"],
            title_of=lambda e: "",
            url_of=lambda e: e.get("url", ""),
            toks_of=lambda e, t: e.get("token_count") or len(t) // 4,  # exact GPT-2 count
            out=args.out_dir / "fineweb_edu.jsonl",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki-tokens", type=int, default=120_000_000)
    ap.add_argument("--fineweb-tokens", type=int, default=120_000_000)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="FineWeb-Edu quality score floor (0 = keep all)")
    ap.add_argument("--out-dir", type=Path, default=DATA_RAW)
    args = ap.parse_args()
    crawl(args)


if __name__ == "__main__":
    main()
