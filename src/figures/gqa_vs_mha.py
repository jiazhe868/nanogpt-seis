"""GQA vs MHA: how grouped-query attention shrinks the KV heads / cache."""
from __future__ import annotations

from pathlib import Path

from ._style import (AQUA, BLUE, INK, INK2, MUTED, PANEL, RED, VIOLET, apply_base,
                     save)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"

GROUP_COLORS = [BLUE, AQUA, VIOLET, RED]     # 4 groups of 3 query heads


def sq(ax, cx, cy, s, color, label, fill=None, fs=8):
    ax.add_patch(FancyBboxPatch((cx - s / 2, cy - s / 2), s, s,
                 boxstyle="round,pad=0.01,rounding_size=0.05",
                 linewidth=1.6, edgecolor=color, facecolor=fill or (color + "30")))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=INK)


def line(ax, x1, y1, x2, y2, color):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-",
                 lw=1.4, color=color, alpha=0.8, shrinkA=1, shrinkB=1))


def panel(ax, y0, title, n_kv, subtitle):
    n_q = 12
    size = 0.62
    x0, step = 2.2, 1.05
    xq = [x0 + i * step for i in range(n_q)]
    yq, ykv = y0 + 1.7, y0
    ax.text(1.7, yq + 0.95, title, fontsize=12.5, fontweight="bold", color=INK)
    ax.text(1.7, yq + 0.55, subtitle, fontsize=9, color=INK2)
    # query heads (always 12, grouped in 4 colours)
    for i, x in enumerate(xq):
        group = i // (n_q // 4)
        sq(ax, x, yq, size, GROUP_COLORS[group], f"Q{i+1}", fs=7)
    # kv heads
    per = n_q // n_kv
    for j in range(n_kv):
        # centre the KV head under the query heads it serves
        served = xq[j * per:(j + 1) * per]
        cxkv = sum(served) / len(served)
        col = GROUP_COLORS[(j * per) // (n_q // 4)]   # colour by the group it serves
        sq(ax, cxkv, ykv, size, col, "KV", fill=col + "45", fs=7)
        for x in served:
            line(ax, x, yq - size / 2, cxkv, ykv + size / 2, col)
    ax.text(xq[-1] + 1.1, yq, "queries", fontsize=8.5, color=MUTED, va="center")
    ax.text(xq[-1] + 1.1, ykv, "keys/values", fontsize=8.5, color=MUTED, va="center")


def main():
    apply_base()
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    ax.set_xlim(0, 16.5); ax.set_ylim(0, 12.2); ax.axis("off")
    ax.text(0.2, 11.7, "Grouped-Query Attention (GQA) vs Multi-Head Attention (MHA)",
            fontsize=15, fontweight="bold", color=INK)

    panel(ax, 7.2, "MHA — 12 query · 12 KV heads",
          12, "every query head has its own K/V  →  largest KV cache")
    panel(ax, 2.4, "GQA — 12 query · 4 KV heads  (this model)",
          4, "each K/V head is shared by 3 query heads  →  3× smaller KV cache")

    note = ("GQA interpolates between MHA (n_kv = n_head, full quality, full cache) and "
            "MQA (n_kv = 1, tiny cache, some quality loss).\n"
            "With n_kv = 4 the per-token KV cache is 3× smaller than MHA — cheaper "
            "long-context inference — at nearly no quality cost,\n"
            "because attention quality is dominated by the number of QUERY heads, which stays 12.")
    ax.add_patch(FancyBboxPatch((0.4, 0.15), 15.7, 1.55,
                 boxstyle="round,pad=0.03,rounding_size=0.05",
                 linewidth=1.3, edgecolor=MUTED, facecolor=PANEL))
    ax.text(0.65, 0.92, note, fontsize=9.2, color=INK2, va="center")

    save(fig, str(ASSETS / "gqa_vs_mha"))
    print("saved assets/gqa_vs_mha.{png,pdf}")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    main()
