"""Generate text from a trained checkpoint.

Usage:
  python -m src.sample --prompt "The 2011 Tohoku earthquake" --num-samples 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model.gqa_gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
TOKENIZED = ROOT / "data" / "tokenized"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "ckpt.pt")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tok = Tokenizer.from_file(str(TOKENIZED / "tokenizer.json"))
    meta = json.loads((TOKENIZED / "meta.json").read_text())
    eot_id = meta["eot_id"]

    ck = torch.load(args.ckpt, map_location=device)
    mcfg = ck["model_cfg"]
    gptcfg = GPTConfig(**{k: mcfg[k] for k in GPTConfig.__dataclass_fields__ if k in mcfg})
    model = GPT(gptcfg)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    print(f"[sample] loaded {args.ckpt} (iter {ck['iter_num']}, val {ck['best_val']:.3f})")

    # Empty prompt -> start a fresh document from the eot token.
    if args.prompt:
        start_ids = tok.encode(args.prompt).ids
    else:
        start_ids = [eot_id]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    for i in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                           top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                           no_repeat_ngram=args.no_repeat_ngram)
        out_ids = [t for t in y[0].tolist() if t != eot_id]
        print(f"\n===== sample {i + 1} =====")
        print(tok.decode(out_ids))


if __name__ == "__main__":
    main()
