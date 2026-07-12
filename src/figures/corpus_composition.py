"""Corpus composition — token share by source, domain vs general."""
from __future__ import annotations

import collections
import json
from pathlib import Path

from matplotlib.patches import Patch

from ._style import AQUA, BLUE, GRID, INK2, apply_base, save
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
ASSETS = ROOT / "assets"
GENERAL = {"wikipedia", "fineweb_edu"}
TOK_PER_WORD = 822.7 / 485.7          # measured: 823M BPE tokens / 485.7M words


def main():
    words = collections.Counter()
    for split in ("train", "val"):
        with (PROC / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                words[rec["source"]] += len(rec["text"].split())

    items = [(src, count) for src, count in words.most_common() if count > 0]
    labels = [src for src, _ in items]
    toks = [count * TOK_PER_WORD / 1e6 for _, count in items]   # million tokens
    colors = [AQUA if s in GENERAL else BLUE for s in labels]

    gen = sum(c for s, c in words.items() if s in GENERAL)
    dom = sum(words.values()) - gen
    ratio = gen / max(1, dom)

    apply_base()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, toks, color=colors, height=0.72)
    for yi, t in zip(y, toks):
        ax.text(t + max(toks) * 0.012, yi, f"{t:.0f}M", va="center",
                fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("tokens (millions, estimated)")
    ax.set_xlim(0, max(toks) * 1.12)
    ax.set_title(f"Corpus composition — 485.7M words ≈ 823M tokens  "
                 f"({gen/(gen+dom)*100:.0f}% general : {dom/(gen+dom)*100:.0f}% domain, "
                 f"{ratio:.1f}:1)", loc="left", fontsize=12)
    ax.legend(handles=[Patch(color=AQUA, label="general (fluency)"),
                       Patch(color=BLUE, label="earthquake domain")],
              frameon=False, loc="lower right")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ASSETS.mkdir(exist_ok=True)
    save(fig, str(ASSETS / "corpus_composition"))
    print("saved assets/corpus_composition.{png,pdf}")


if __name__ == "__main__":
    main()
