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

from ._style import (AXIS, BLUE, GRID, INK2, ORANGE, RED, apply_base, save)
import matplotlib.pyplot as plt

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
    ratio = (it - WARMUP) / (MAX_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * ratio)) * (LR - MIN_LR)


def parse():
    txt = LOG.read_text(errors="ignore")
    train_rows = re.findall(r"\[train\] iter (\d+): loss ([\d.]+)", txt)
    eval_rows = re.findall(r"\[eval\] iter (\d+): train ([\d.]+) val ([\d.]+)", txt)
    t_it = np.array([int(it) for it, _ in train_rows])
    t_ls = np.array([float(loss) for _, loss in train_rows])
    e_it = np.array([int(it) for it, _, _ in eval_rows])
    e_va = np.array([float(val) for _, _, val in eval_rows])
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
    best_i = int(np.argmin(e_va))
    ax.scatter([e_it[best_i]], [e_va[best_i]], s=80, facecolor="none", edgecolor=RED,
               lw=2, zorder=5)
    ax.annotate(f"best val {e_va[best_i]:.3f}  (ppl {math.exp(e_va[best_i]):.1f})",
                xy=(e_it[best_i], e_va[best_i]),
                xytext=(e_it[best_i] - 2600, e_va[best_i] + 0.9),
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
    save(fig, str(ASSETS / "training_dynamics"))
    print("saved assets/training_dynamics.{png,pdf}")


def fig_lr_vs_loss(t_it, t_ls):
    """Dual-axis over training step: learning rate (left) and loss (right)."""
    apply_base()
    lrs = np.array([get_lr(i) for i in t_it])
    loss = ema(t_ls, 0.06)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax2 = ax.twinx()

    l_lr, = ax.plot(t_it, lrs, color=ORANGE, lw=2.2, label="learning rate")
    l_loss, = ax2.plot(t_it, loss, color=BLUE, lw=2.2, label="loss (train, EMA)")

    # left axis = learning rate (label coloured to its curve; spine stays gray)
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate", color=ORANGE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlim(0, t_it.max())
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    # right axis = loss; a twin re-enables the right spine (the loss axis), kept
    # thin gray to match the house style.
    ax2.set_ylabel("cross-entropy loss", color=BLUE)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(AXIS)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.set_ylim(2.0, 10)

    ax.set_title("Learning-rate schedule and loss over training",
                 loc="left", fontsize=13)
    ax.legend(handles=[l_lr, l_loss], frameon=False, loc="upper right")
    save(fig, str(ASSETS / "lr_vs_loss"))
    print("saved assets/lr_vs_loss.{png,pdf}")


def main():
    ASSETS.mkdir(exist_ok=True)
    t_it, t_ls, e_it, e_va = parse()
    print(f"parsed {len(t_it)} train points, {len(e_it)} eval points")
    fig_dynamics(t_it, t_ls, e_it, e_va)
    fig_lr_vs_loss(t_it, t_ls)


if __name__ == "__main__":
    main()
