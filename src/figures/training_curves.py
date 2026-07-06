"""Densely-sampled training dynamics + learning-rate-vs-loss, from the 4k log.

Reads per-10-iter train loss (800 points) from checkpoints/train_ctx4k.log and
the 17 eval points, and reconstructs the LR schedule analytically.

  python -m src.figures.training_curves
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ._style import AQUA, BLUE, GRID, INK, INK2, RED, SURFACE, apply_base
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "checkpoints" / "train_mix.log"      # current model: general+domain mix
ASSETS = ROOT / "assets"

# schedule (configs/gpt120m_ctx4k.yaml)
WARMUP, MAX_ITERS, LR, MIN_LR = 400, 8000, 3.0e-4, 3.0e-5


def get_lr(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    if it >= MAX_ITERS:
        return MIN_LR
    r = (it - WARMUP) / (MAX_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)


def parse():
    txt = LOG.read_text(errors="ignore")
    tr = re.findall(r"\[train\] iter (\d+): loss ([\d.]+)", txt)
    ev = re.findall(r"\[eval\] iter (\d+): train ([\d.]+) val ([\d.]+)", txt)
    t_it = np.array([int(a) for a, _ in tr])
    t_ls = np.array([float(b) for _, b in tr])
    e_it = np.array([int(a) for a, _, _ in ev])
    e_va = np.array([float(c) for _, _, c in ev])
    return t_it, t_ls, e_it, e_va


def ema(x, alpha=0.1):
    out = np.empty_like(x, dtype=float)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def fig_dynamics(t_it, t_ls, e_it, e_va):
    apply_base()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(t_it, t_ls, color=BLUE, lw=0.8, alpha=0.30, label="train (per 10 iters)")
    ax.plot(t_it, ema(t_ls, 0.06), color=BLUE, lw=2.2, label="train (EMA)")
    ax.plot(e_it, e_va, color=RED, lw=2, marker="o", ms=4, label="val (per 500 iters)")
    bi = int(np.argmin(e_va))
    ax.scatter([e_it[bi]], [e_va[bi]], s=80, facecolor="none", edgecolor=RED, lw=2, zorder=5)
    ax.annotate(f"best val {e_va[bi]:.3f}  (ppl {math.exp(e_va[bi]):.1f})",
                xy=(e_it[bi], e_va[bi]), xytext=(e_it[bi] - 2600, e_va[bi] + 0.9),
                fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_yscale("log")
    ax.set_yticks([2, 2.5, 3, 4, 6, 9])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(1.9, 10)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss (log scale)")
    ax.set_title("nanoGPT-Seis — training dynamics (113M, ctx 4096, general+domain mix)",
                 loc="left", fontsize=13)
    ax.grid(True, which="both", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(ASSETS / "training_dynamics.png", dpi=150, bbox_inches="tight")
    print("saved assets/training_dynamics.png")


def fig_lr_vs_loss(t_it, t_ls):
    apply_base()
    lrs = np.array([get_lr(i) for i in t_it])
    sm = ema(t_ls, 0.06)
    # colour the trajectory by iteration so warmup vs decay is legible
    pts = np.column_stack([lrs, sm]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    lc = LineCollection(segs, cmap="viridis", array=t_it[:-1], lw=2.4)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax)
    cb.set_label("iteration", color=INK2)
    # mark warmup peak
    pk = int(np.argmax(lrs))
    ax.scatter([lrs[pk]], [sm[pk]], s=60, color=RED, zorder=5)
    ax.annotate("end of warmup\n(peak LR)", xy=(lrs[pk], sm[pk]),
                xytext=(lrs[pk] * 0.55, sm[pk] + 1.6), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.annotate("warmup:\nLR↑, loss crashes", xy=(1.1e-4, 6.0), fontsize=9, color=INK2)
    ax.annotate("cosine decay:\nLR↓, loss slowly refines", xy=(0.4e-4, 2.5),
                fontsize=9, color=INK2)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("cross-entropy loss (EMA)")
    ax.set_yscale("log"); ax.set_ylim(1.9, 10)
    ax.set_yticks([2, 2.5, 3, 4, 6, 9])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0, LR * 1.05)
    ax.set_title("Learning rate vs. loss (warmup → cosine decay)", loc="left", fontsize=13)
    ax.grid(True, which="both", color=GRID, lw=0.9); ax.set_axisbelow(True)
    fig.savefig(ASSETS / "lr_vs_loss.png", dpi=150, bbox_inches="tight")
    print("saved assets/lr_vs_loss.png")


def main():
    ASSETS.mkdir(exist_ok=True)
    t_it, t_ls, e_it, e_va = parse()
    print(f"parsed {len(t_it)} train points, {len(e_it)} eval points")
    fig_dynamics(t_it, t_ls, e_it, e_va)
    fig_lr_vs_loss(t_it, t_ls)


if __name__ == "__main__":
    main()
