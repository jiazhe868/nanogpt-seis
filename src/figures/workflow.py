"""Overall pipeline / training workflow diagram (single-row pipeline)."""
from __future__ import annotations

from pathlib import Path

from ._style import (AQUA, BLUE, GREEN, INK, INK2, MUTED, ORANGE, RED,
                     VIOLET, YELLOW, apply_base)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=2, edgecolor=color, facecolor=color + "18"))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - i * 0.40, ln, ha="center", va="top",
                fontsize=8.8, color=INK2)


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0))


def main():
    apply_base()
    fig, ax = plt.subplots(figsize=(17, 5.6))
    ax.set_xlim(0, 25.4); ax.set_ylim(0, 8.4); ax.axis("off")
    ax.text(0.1, 8.0, "nanoGPT-Seis — end-to-end pretraining pipeline",
            fontsize=15, fontweight="bold", color=INK)

    stages = [
        ("1. Crawl", ["concurrent, resumable", "PDF→text + validate"], BLUE, "data/raw/*.jsonl"),
        ("2. Process", ["clean · filter", "MinHash dedup · split"], AQUA, "processed/*.jsonl"),
        ("3. Tokenize", ["byte-level BPE 16k", "encode → uint16"], YELLOW, "tokenized/*.bin"),
        ("4. Model", ["113M GQA + RoPE", "RMSNorm · SwiGLU"], VIOLET, "src/model/gqa_gpt.py"),
        ("5. Train", ["2×A30 DDP · bf16", "cosine LR · compile"], RED, "checkpoints/ckpt.pt"),
        ("6. Inference", ["KV-cache streaming", "perplexity · figures"], GREEN, "sample / inference"),
    ]
    w, h, gap = 3.55, 3.0, 0.62
    y = 2.6
    x = 0.1
    centers = []
    for title, lines, c, art in stages:
        box(ax, x, y, w, h, title, lines, c)
        ax.text(x + w / 2, y - 0.45, art, ha="center", va="center", fontsize=8,
                color=MUTED, family="monospace")
        centers.append((x, x + w))
        x += w + gap
    for i in range(len(stages) - 1):
        arrow(ax, centers[i][1], y + h / 2, centers[i + 1][0], y + h / 2)

    # data sources feeding stage 1 from above (domain + general mix)
    sources = [("arXiv", BLUE), ("Crossref +\nUnpaywall", AQUA), ("EarthArXiv", VIOLET),
               ("Substack", RED), ("Wikipedia", ORANGE), ("FineWeb-Edu", GREEN)]
    sx = 0.1
    cw = 1.5
    ax.text(0.1, 7.15, "free data sources — earthquake domain  +  general (fluency)",
            fontsize=9.5, color=MUTED, style="italic")
    for name, c in sources:
        ax.add_patch(FancyBboxPatch((sx, 6.0), cw, 0.85,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     linewidth=1.4, edgecolor=c, facecolor=c + "20"))
        ax.text(sx + cw / 2, 6.42, name, ha="center", va="center", fontsize=7.6, color=INK)
        sx += cw + 0.12
    # converge arrow into Crawl
    arrow(ax, (0.1 + sx) / 2 - 0.3, 6.0, centers[0][0] + w / 2, y + h, color=MUTED)

    fig.savefig(ASSETS / "workflow.png", dpi=150, bbox_inches="tight")
    print("saved assets/workflow.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    main()
