"""IsoFLOP profiles + the compute-optimal frontier.

Left : loss vs model size (one curve per compute budget), with the fitted parabola
       and its minimum marked — the compute-optimal size for that budget.
Right: N_opt vs C on log-log axes, with the fitted power law N_opt ∝ C^a.

Run `python -m src.scaling.fit` first to produce scaling_fit.json.

  python -m src.figures.scaling_laws
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ._style import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, RED, VIOLET,
                     apply_base, save)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
FIT = ROOT / "checkpoints" / "scaling" / "scaling_fit.json"
BUDGET_COLORS = [BLUE, ORANGE, GREEN, RED, VIOLET]   # one per budget (up to 5)


def main():
    if not FIT.exists():
        raise SystemExit(f"{FIT} missing — run `python -m src.scaling.fit` first")
    data = json.loads(FIT.read_text())
    runs, frontier = data["runs"], data["frontier"]

    by_budget = defaultdict(list)
    for r in runs:
        by_budget[r["budget"]].append(r)

    apply_base()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # --- Panel A: IsoFLOP profiles (loss vs N per budget) ---
    for k, (C, pts) in enumerate(sorted(by_budget.items())):
        col = BUDGET_COLORS[k % len(BUDGET_COLORS)]
        pts = sorted(pts, key=lambda p: p["N"])
        Ns = np.array([p["N"] for p in pts], dtype=float)
        Ls = np.array([p["loss"] for p in pts], dtype=float)
        axL.scatter(Ns, Ls, s=42, color=col, zorder=3, label=f"C = {C:.0e} FLOPs")
        if len(pts) >= 3:
            logN = np.log(Ns)
            a, b, c = np.polyfit(logN, Ls, 2)
            xs = np.linspace(logN.min(), logN.max(), 100)
            axL.plot(np.exp(xs), a * xs**2 + b * xs + c, color=col, lw=1.6, alpha=0.7)
        fr = next((f for f in frontier if f["budget"] == C), None)
        if fr:
            axL.scatter([fr["N_opt"]], [fr["L_opt"]], s=120, facecolor="none",
                        edgecolor=col, lw=2.2, zorder=4)
    axL.set_xscale("log")
    axL.set_xlabel("non-embedding parameters  N")
    axL.set_ylabel("validation loss")
    axL.set_title("IsoFLOP profiles — the loss valley per compute budget\n"
                  "(circle = compute-optimal size)", loc="left", fontsize=11)
    axL.grid(True, which="both", color=GRID, lw=0.7); axL.set_axisbelow(True)
    axL.legend(frameon=False, loc="upper center")

    # --- Panel B: the compute-optimal frontier N_opt(C) ---
    if len(frontier) >= 2:
        Cs = np.array([f["budget"] for f in frontier])
        Ns = np.array([f["N_opt"] for f in frontier])
        ref = data.get("reference")
        axR.scatter(Cs, Ns, s=60, color=INK, zorder=3, label="compute-optimal (fit)")
        exps = data.get("exponents", {})
        exponent = exps.get("N_vs_C")
        if exponent is not None:
            coef = exps["kN"]
            c_hi = max(Cs.max(), ref["C"]) if ref else Cs.max()
            xs = np.logspace(math.log10(Cs.min()), math.log10(c_hi), 200)
            in_range = xs <= Cs.max()
            axR.plot(xs[in_range], coef * xs[in_range] ** exponent, color=ORANGE, lw=2,
                     label=f"N_opt ~ C^{exponent:.2f}")
            if ref and (~in_range).any():                 # extrapolation past the sweep
                axR.plot(xs[~in_range], coef * xs[~in_range] ** exponent, color=ORANGE,
                         lw=2, ls="--", alpha=0.8, label="frontier (extrapolated)")
        if ref:
            axR.scatter([ref["C"]], [ref["N"]], s=200, marker="*", color=RED, zorder=5,
                        label=ref["name"])
            ratio = ref.get("ratio_N_over_Nopt")
            if ratio:
                axR.annotate(f"{ratio:.1f}× N_opt", xy=(ref["C"], ref["N"]),
                             xytext=(ref["C"] * 0.4, ref["N"] * 1.7),
                             fontsize=9, color=INK)
        axR.set_xscale("log"); axR.set_yscale("log")
        axR.set_xlabel("compute  C  (FLOPs)")
        axR.set_ylabel("compute-optimal size  N_opt")
        axR.set_title("Compute-optimal frontier + the real 113M run\n"
                      "(Chinchilla predicts an exponent ≈ 0.5)", loc="left", fontsize=11)
        axR.grid(True, which="both", color=GRID, lw=0.7); axR.set_axisbelow(True)
        axR.legend(frameon=False, loc="upper left", fontsize=8.5)
    else:
        axR.text(0.5, 0.5, "need ≥2 finished budgets\nto draw the frontier",
                 ha="center", va="center", color=MUTED, transform=axR.transAxes)
        axR.axis("off")

    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    save(fig, str(ASSETS / "scaling_laws"))
    print("saved assets/scaling_laws.{png,pdf}")


if __name__ == "__main__":
    main()
