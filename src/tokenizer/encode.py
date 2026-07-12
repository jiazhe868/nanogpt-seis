"""Encode the processed corpus to memory-mappable uint16 token shards.

nanoGPT layout: one flat array of token ids per split, documents separated by
the <|endoftext|> id. Training reads these with np.memmap and samples random
context windows.

Usage:
  python -m src.tokenizer.encode
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
TOKENIZED = Path(__file__).resolve().parents[2] / "data" / "tokenized"

FLUSH_EVERY = 2_000_000       # token ids buffered before writing to disk
BATCH_DOCS = 1000             # docs per encode_batch call


def _docs(jsonl: Path):
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]


def encode_split(tok: Tokenizer, jsonl: Path, out_bin: Path, eot_id: int) -> int:
    """Encode one split to `out_bin`. Returns total token count."""
    total = 0
    buf: list[int] = []
    batch: list[str] = []

    def flush_encode(texts: list[str]) -> None:
        if not texts:
            return
        for enc in tok.encode_batch(texts):
            buf.extend(enc.ids)
            buf.append(eot_id)          # doc separator

    with out_bin.open("wb") as fout:
        for text in _docs(jsonl):
            batch.append(text)
            if len(batch) >= BATCH_DOCS:
                flush_encode(batch)
                batch = []
            if len(buf) >= FLUSH_EVERY:
                arr = np.asarray(buf, dtype=np.uint16)
                arr.tofile(fout)
                total += len(buf)
                buf.clear()
        flush_encode(batch)             # tail
        if buf:
            arr = np.asarray(buf, dtype=np.uint16)
            arr.tofile(fout)
            total += len(buf)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZED / "tokenizer.json")
    args = parser.parse_args()

    meta = json.loads((TOKENIZED / "meta.json").read_text())
    eot_id = meta["eot_id"]
    assert meta["vocab_size"] <= 65536, "vocab too large for uint16"
    tok = Tokenizer.from_file(str(args.tokenizer))

    counts = {}
    for split in ("train", "val"):
        src = PROCESSED / f"{split}.jsonl"
        if not src.exists():
            continue
        out_bin = TOKENIZED / f"{split}.bin"
        n = encode_split(tok, src, out_bin, eot_id)
        counts[split] = n
        print(f"[encode] {split}: {n:,} tokens -> {out_bin} "
              f"({out_bin.stat().st_size/1e6:.1f} MB)")

    meta["token_counts"] = counts
    meta["dtype"] = "uint16"
    (TOKENIZED / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[encode] done. {sum(counts.values()):,} tokens total")


if __name__ == "__main__":
    main()
