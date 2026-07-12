"""Train a byte-level BPE tokenizer on the processed corpus (train split only).

Byte-level => every possible byte is in the base alphabet, so there are no
unknown tokens for any input. We add a single special token, <|endoftext|>,
used to separate documents at encode time.

Usage:
  python -m src.tokenizer.train_bpe --vocab-size 16384
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
TOKENIZED = Path(__file__).resolve().parents[2] / "data" / "tokenized"

EOT = "<|endoftext|>"


def _iter_texts(jsonl: Path):
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--train", type=Path, default=PROCESSED / "train.jsonl")
    parser.add_argument("--out", type=Path, default=TOKENIZED / "tokenizer.json")
    args = parser.parse_args()

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[EOT],
        initial_alphabet=ByteLevel.alphabet(),   # all 256 byte reprs
        show_progress=True,
    )

    print(f"[bpe] training on {args.train} (vocab={args.vocab_size}) ...")
    tokenizer.train_from_iterator(_iter_texts(args.train), trainer=trainer)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))
    eot_id = tokenizer.token_to_id(EOT)
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eot_token": EOT,
        "eot_id": eot_id,
        "tokenizer": args.out.name,
    }
    (args.out.parent / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[bpe] saved -> {args.out}")
    print(f"[bpe] vocab_size={meta['vocab_size']}  eot_id={eot_id}")


if __name__ == "__main__":
    main()
