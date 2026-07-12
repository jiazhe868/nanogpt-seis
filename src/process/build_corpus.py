"""Phase 2 orchestrator: raw JSONL -> cleaned, filtered, deduped train/val.

  data/raw/*.jsonl
      -> clean (source-aware)  -> filter (len / english / alpha)
      -> dedup (by id, then exact, then MinHash near-dup)
      -> shuffle + split        -> data/processed/{train,val}.jsonl + stats.json

Usage:
  python -m src.process.build_corpus --val-frac 0.01
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from .clean import clean_doc, passes_filters
from .dedup import dedup

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_raw() -> list[dict]:
    docs: list[dict] = []
    for fp in sorted(RAW.glob("*.jsonl")):
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--near-dup-threshold", type=float, default=0.7)
    args = parser.parse_args()

    raw = load_raw()
    print(f"[build] loaded {len(raw)} raw docs from {RAW}")
    if not raw:
        print("[build] no raw docs found — run the crawlers first."); return

    # 1) clean + filter, dedup by work id keeping the longest text so that a
    #    paper present as both an abstract and a full-text PDF collapses to the
    #    full-text version.
    filt_reasons: collections.Counter = collections.Counter()
    best_by_id: dict[str, dict] = {}
    for d in raw:
        text = clean_doc(d["source"], d.get("text", ""))
        ok, reason = passes_filters(text, min_chars=args.min_chars)
        if not ok:
            filt_reasons[reason] += 1
            continue
        d["text"] = text
        prev = best_by_id.get(d["id"])
        if prev is None:
            best_by_id[d["id"]] = d
        else:
            filt_reasons["dup_id"] += 1
            if len(text) > len(prev["text"]):
                best_by_id[d["id"]] = d       # keep the fuller version
    cleaned = list(best_by_id.values())
    print(f"[build] after clean+filter: {len(cleaned)}  (dropped: {dict(filt_reasons)})")

    # 2) near-dup removal.
    kept, dstats = dedup(cleaned, threshold=args.near_dup_threshold)
    print(f"[build] dedup: {dstats}")

    # 3) shuffle + split (deterministic).
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_frac))
    val, train = kept[:n_val], kept[n_val:]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val)]:
        with (PROCESSED / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for d in split:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 4) stats.
    total_chars = sum(len(d["text"]) for d in kept)
    total_words = sum(len(d["text"].split()) for d in kept)
    kept_by_source = collections.Counter(d["source"] for d in kept)
    n_fulltext = sum(1 for d in kept if d.get("extra", {}).get("full_text"))
    stats = {
        "raw_docs": len(raw),
        "after_filter": len(cleaned),
        "filter_dropped": dict(filt_reasons),
        "dedup": dstats,
        "final_docs": len(kept),
        "full_text_docs": n_fulltext,
        "train_docs": len(train),
        "val_docs": len(val),
        "final_by_source": dict(kept_by_source),
        "total_chars": total_chars,
        "total_words": total_words,
        "est_tokens_gpt2": round(total_chars / 4),   # ~4 chars/token rule
    }
    (PROCESSED / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[build] wrote train={len(train)} val={len(val)} -> {PROCESSED}")
    print(f"[build] {total_words/1e6:.2f}M words / {total_chars/1e6:.1f}M chars "
          f"(~{stats['est_tokens_gpt2']/1e6:.1f}M tokens); {n_fulltext} full-text docs")
    print(f"[build] by source: {dict(kept_by_source)}")


if __name__ == "__main__":
    main()
