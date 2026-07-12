"""Plot training dynamics from checkpoints/log.csv.

Two stacked panels sharing the iteration axis (never a dual y-axis):
  1. train vs val loss   2. learning-rate schedule

Usage:
  python -m src.plot_training
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Validated categorical hues (dataviz reference palette): slot 1 blue, slot 6 red,
# slot 5 violet. Ink tokens for all text; recessive grid.
C_TRAIN = "#2a78d6"
C_VAL = "#e34948"
C_LR = "#4a3aa7"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e6e3"
SURFACE = "#fcfcfb"


def load(path: Path):
    it, tr, va, lr = [], [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            it.append(int(row["iter"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
            lr.append(float(row["lr"]))
    return it, tr, va, lr


def _best(it, va):
    k = min(range(len(va)), key=lambda j: va[j])
    return it[k], va[k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=ROOT / "checkpoints" / "log.csv")
    parser.add_argument("--label", type=str, default="ctx 4096")
    parser.add_argument("--compare-log", type=Path, default=None,
                    help="optional second run to overlay (e.g. the 1024 baseline)")
    parser.add_argument("--compare-label", type=str, default="ctx 1024")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "training_dynamics.png")
    args = parser.parse_args()

    it, tr, va, lr = load(args.log)
    best_it, best_val = _best(it, va)

    # ---- comparison overlay path ----
    if args.compare_log and args.compare_log.exists():
        cit, ctr, cva, _ = load(args.compare_log)
        cbest_it, cbest_val = _best(cit, cva)
        plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                             "font.size": 11, "axes.edgecolor": INK2, "text.color": INK,
                             "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(it, va, color=C_TRAIN, lw=2, marker="o", ms=3.5, label=f"{args.label} — val")
        ax.plot(it, tr, color=C_TRAIN, lw=1.2, ls="--", alpha=0.6, label=f"{args.label} — train")
        ax.plot(cit, cva, color=C_VAL, lw=2, marker="o", ms=3.5, label=f"{args.compare_label} — val")
        ax.plot(cit, ctr, color=C_VAL, lw=1.2, ls="--", alpha=0.6, label=f"{args.compare_label} — train")
        for (bi, bv, col) in [(best_it, best_val, C_TRAIN), (cbest_it, cbest_val, C_VAL)]:
            ax.scatter([bi], [bv], s=60, facecolor="none", edgecolor=col, lw=2, zorder=5)
        ax.set_ylim(2.0, 4.2)
        ax.set_xlabel("iteration"); ax.set_ylabel("cross-entropy loss")
        ax.set_title("nanoGPT-Seis — context length: 4096 vs 1024",
                     fontsize=13, fontweight="bold", loc="left", color=INK)
        ax.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        ax.annotate(f"best val {best_val:.3f} (ppl {math.exp(best_val):.1f})",
                    xy=(best_it, best_val), xytext=(best_it - 1200, best_val - 0.28),
                    fontsize=9, color=C_TRAIN,
                    arrowprops=dict(arrowstyle="-", color=C_TRAIN, lw=1))
        ax.annotate(f"best val {cbest_val:.3f} (ppl {math.exp(cbest_val):.1f})",
                    xy=(cbest_it, cbest_val), xytext=(cbest_it - 1200, cbest_val + 0.22),
                    fontsize=9, color=C_VAL,
                    arrowprops=dict(arrowstyle="-", color=C_VAL, lw=1))
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
        fig.savefig(str(args.out).replace(".png", ".pdf"), bbox_inches="tight")
        print(f"[plot] comparison saved -> {args.out}")
        print(f"[plot] {args.label} best val {best_val:.4f} (ppl {math.exp(best_val):.2f}) @ {best_it}")
        print(f"[plot] {args.compare_label} best val {cbest_val:.4f} (ppl {math.exp(cbest_val):.2f}) @ {cbest_it}")
        return

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.size": 11, "axes.edgecolor": INK2, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    # --- panel 1: loss ---
    ax1.plot(it, tr, color=C_TRAIN, lw=2, marker="o", ms=4, label="train")
    ax1.plot(it, va, color=C_VAL, lw=2, marker="o", ms=4, label="val")
    # mark best val
    ax1.scatter([best_it], [best_val], s=70, facecolor="none",
                edgecolor=C_VAL, lw=2, zorder=5)
    ax1.annotate(f"best val {best_val:.3f}\n(ppl {math.exp(best_val):.1f}) @ {best_it}",
                 xy=(best_it, best_val), xytext=(best_it, best_val + 1.1),
                 ha="center", fontsize=9, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("nanoGPT-Seis — training dynamics (113M, 2×A30)",
                  fontsize=13, fontweight="bold", loc="left", color=INK)
    ax1.grid(True, color=GRID, lw=1)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="upper right")
    # direct-label the series ends too
    ax1.text(it[-1], tr[-1] - 0.12, "train", color=C_TRAIN, fontsize=9, va="top", ha="right")

    # --- panel 2: learning rate ---
    ax2.plot(it, lr, color=C_LR, lw=2)
    ax2.set_ylabel("learning rate")
    ax2.set_xlabel("iteration")
    ax2.grid(True, color=GRID, lw=1)
    ax2.set_axisbelow(True)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(str(args.out).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"[plot] saved -> {args.out}")
    print(f"[plot] best val {best_val:.4f} (ppl {math.exp(best_val):.2f}) at iter {best_it}")


if __name__ == "__main__":
    main()
