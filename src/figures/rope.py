"""What RoPE encodes — two views (this model: head_dim 64, theta 10000).

Left : the q·k score depends only on the *relative* distance (query − key), so
       curves computed at different absolute positions lie on top of each other.
Right: the multi-scale rotation — each channel pair rotates at its own frequency,
       fast (top) to slow (bottom).
No trained weights needed: RoPE is parameter-free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps

from ._style import BLUE, GRID, INK2, ORANGE, VIOLET, apply_base, save
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
HEAD_DIM = 64
THETA = 10000.0
INV_FREQ = 1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))   # (32,)


def rope_apply(v, pos):
    """Rotate a single head vector v (head_dim,) to position `pos` (rotate_half)."""
    emb = np.concatenate([pos * INV_FREQ, pos * INV_FREQ])           # (64,)
    x1, x2 = v[:HEAD_DIM // 2], v[HEAD_DIM // 2:]
    return v * np.cos(emb) + np.concatenate([-x2, x1]) * np.sin(emb)


def main():
    apply_base()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # --- Panel A: relative-distance property ---
    base_vec = np.random.default_rng(0).standard_normal(HEAD_DIM)
    norm = base_vec @ base_vec
    rel_dist = np.arange(0, 257)
    styles = [(200, BLUE, "-", 2.4), (800, ORANGE, "--", 1.8), (2000, VIOLET, ":", 1.8)]
    for query_pos, color, ls, lw in styles:
        q_rot = rope_apply(base_vec, query_pos)
        sims = np.array([q_rot @ rope_apply(base_vec, query_pos - d) for d in rel_dist]) / norm
        axL.plot(rel_dist, sims, color=color, ls=ls, lw=lw,
                 label=f"query at position {query_pos}")
    axL.axhline(0, color=INK2, lw=0.6)
    axL.set_xlabel("relative distance  (query − key position)")
    axL.set_ylabel("normalized q·k  (relative similarity)")
    axL.set_title("RoPE is relative — q·k depends only on the distance;\n"
                  "curves at different absolute positions coincide", loc="left", fontsize=11)
    axL.grid(True, color=GRID, lw=0.8); axL.set_axisbelow(True)
    axL.legend(frameon=False, loc="upper right")

    # --- Panel B: multi-scale rotation spectrum ---
    P = 256
    cos = np.cos(np.outer(INV_FREQ, np.arange(P)))                   # (32, P)
    im = axR.imshow(cos, aspect="auto", cmap=colormaps["RdBu_r"], vmin=-1, vmax=1,
                    extent=[0, P, HEAD_DIM // 2, 0])
    axR.set_xlabel("token position")
    axR.set_ylabel("channel-pair index  (0 = high freq → 31 = low freq)")
    axR.set_title("Multi-scale rotation — each channel pair spins at its own\n"
                  "frequency (fast at top, slow at bottom)", loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)
    cb.set_label("cos(position × frequency)")

    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    save(fig, str(ASSETS / "rope"))
    print("saved assets/rope.{png,pdf}")


if __name__ == "__main__":
    main()
