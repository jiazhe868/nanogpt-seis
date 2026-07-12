"""Bits-per-byte A/B — domain-only base (v1) vs general+domain mix (v2).

Tokenizer-independent, so the two models (different vocabularies) compare fairly.
Lower = better. Shows the fluency win on general text and the domain-dilution cost.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._style import BLUE, GRID, INK2, MUTED, apply_base, save
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"


def main():
    from src.inference import InferenceEngine
    from src.compare_models import held_out, mean_bpb
    v2 = InferenceEngine(ROOT / "checkpoints" / "ckpt.pt")
    v1 = InferenceEngine(
        ROOT / "checkpoints" / "ckpt_v1_domain.pt",
        tokenizer_path=ROOT / "data" / "tokenized" / "tokenizer_v1_domain.json",
        meta_path=ROOT / "data" / "tokenized" / "meta_v1_domain.json")

    gen, dom = held_out({"wikipedia", "fineweb_edu"}, 8), held_out({"fulltext", "arxiv"}, 8)
    groups = ["general\n(Wikipedia / web)", "earthquake\npapers"]
    v1v = [mean_bpb(v1, gen), mean_bpb(v1, dom)]
    v2v = [mean_bpb(v2, gen), mean_bpb(v2, dom)]

    apply_base()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(groups)); w = 0.36
    b1 = ax.bar(x - w / 2, v1v, w, label="v1 — domain-only", color=MUTED)
    b2 = ax.bar(x + w / 2, v2v, w, label="v2 — general + domain", color=BLUE)
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9, color=INK2)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9, color=INK2)
    for i in range(len(groups)):                       # % change annotation
        delta = (v2v[i] - v1v[i]) / v1v[i] * 100
        ax.text(x[i], max(v1v[i], v2v[i]) + 0.14, f"{delta:+.0f}%",
                ha="center", fontsize=9.5, fontweight="bold",
                color=("#1b7a3d" if delta < 0 else "#b23b3b"))
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("bits per byte  (lower = better)")
    ax.set_ylim(0, max(v1v + v2v) * 1.25)
    ax.set_title("Fluency A/B: adding general text cuts bits/byte on general prose\n"
                 "(−35%) at a small domain cost — the fluency↔specialization trade-off",
                 loc="left", fontsize=11.5)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ASSETS.mkdir(exist_ok=True)
    save(fig, str(ASSETS / "bpb_comparison"))
    print(f"saved assets/bpb_comparison.{{png,pdf}}  v1={v1v} v2={v2v}")


if __name__ == "__main__":
    main()
