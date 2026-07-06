"""A/B compare the domain-only base (v1) vs the general+domain mix (v2).

v1 and v2 use different tokenizers, so token-perplexity isn't comparable — we use
bits-per-byte (tokenizer-independent). We also generate from both on a general
and an earthquake prompt to see the fluency difference directly.

  CUDA_VISIBLE_DEVICES=0 python -m src.compare_models
"""
from __future__ import annotations

import json
from pathlib import Path

from .inference import InferenceEngine

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data" / "processed" / "val.jsonl"


def held_out(sources, n, max_chars=3000):
    out = []
    for line in VAL.open(encoding="utf-8"):
        d = json.loads(line)
        if d["source"] in sources and len(d["text"]) > 800:
            out.append(d["text"][:max_chars])
            if len(out) >= n:
                break
    return out


def mean_bpb(eng, texts):
    return sum(eng.bits_per_byte(t) for t in texts) / len(texts)


def main():
    v2 = InferenceEngine(ROOT / "checkpoints" / "ckpt.pt")
    v1 = InferenceEngine(
        ROOT / "checkpoints" / "ckpt_v1_domain.pt",
        tokenizer_path=ROOT / "data" / "tokenized" / "tokenizer_v1_domain.json",
        meta_path=ROOT / "data" / "tokenized" / "meta_v1_domain.json",
    )
    print("v1 = domain-only base | v2 = general+domain mix\n")

    gen_txt = held_out({"wikipedia", "fineweb_edu"}, 8)
    dom_txt = held_out({"fulltext", "arxiv"}, 8)
    print("=== bits-per-byte (LOWER = better; tokenizer-independent) ===")
    print(f"{'held-out set':<22}{'v1 (domain)':>14}{'v2 (mix)':>12}")
    for name, txt in [("general (wiki/web)", gen_txt), ("earthquake papers", dom_txt)]:
        b1, b2 = mean_bpb(v1, txt), mean_bpb(v2, txt)
        print(f"{name:<22}{b1:>14.3f}{b2:>12.3f}   ({(1-b2/b1)*100:+.0f}% v2)")

    print("\n=== generations (temp 0.8, top-k 200) ===")
    for prompt in ["The history of the Roman Empire",
                   "The 2011 Tohoku earthquake"]:
        print(f"\n--- prompt: {prompt!r} ---")
        for tag, eng in [("v1", v1), ("v2", v2)]:
            txt = eng.generate(prompt, max_new_tokens=90, temperature=0.8,
                               top_k=200, seed=0)
            print(f"[{tag}] {txt}\n")


if __name__ == "__main__":
    main()
