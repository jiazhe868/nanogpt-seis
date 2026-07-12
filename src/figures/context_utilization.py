"""Context utilization — mean next-token loss by position within a 4096 window.

If the model uses long context, loss should fall for later positions (more
preceding tokens to condition on).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ._style import BLUE, GRID, ORANGE, apply_base, save
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
TOK = ROOT / "data" / "tokenized"


def main(n_windows=80):
    from src.inference import InferenceEngine
    eng = InferenceEngine(ROOT / "checkpoints" / "ckpt.pt")
    block = eng.cfg.block_size
    data = np.memmap(TOK / "val.bin", dtype=np.uint16, mode="r")
    pos_sum = torch.zeros(block, device=eng.device)
    rng = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _ in range(n_windows):
            i = int(torch.randint(len(data) - block - 1, (1,), generator=rng))
            x = torch.from_numpy(data[i:i+block].astype(np.int64)).to(eng.device)[None]
            y = torch.from_numpy(data[i+1:i+1+block].astype(np.int64)).to(eng.device)[None]
            with eng.amp:
                logits, _ = eng.model(x, y)
            pos_sum += F.cross_entropy(logits[0].float(), y[0], reduction="none")
    loss = (pos_sum / n_windows).cpu().numpy()

    # smooth with a rolling mean for readability
    smooth_win = 64
    smoothed = np.convolve(loss, np.ones(smooth_win) / smooth_win, mode="valid")
    xs = np.arange(len(smoothed)) + smooth_win // 2

    apply_base()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(np.arange(block), loss, color=BLUE, lw=0.6, alpha=0.25, label="per position")
    ax.plot(xs, smoothed, color=BLUE, lw=2.2, label=f"rolling mean ({smooth_win})")
    early = loss[:64].mean(); late = loss[2048:].mean()
    ax.axhline(early, color=ORANGE, lw=1, ls="--")
    ax.axhline(late, color="#1b7a3d", lw=1, ls="--")
    ax.text(block * 0.45, early + 0.12, f"positions 0–64: {early:.2f}", color=ORANGE, fontsize=9)
    ax.text(block * 0.45, late - 0.35, f"positions 2048–4096: {late:.2f}  "
            f"(−{(1-late/early)*100:.0f}%)", color="#1b7a3d", fontsize=9)
    ax.set_xlabel("position in context window")
    ax.set_ylabel("mean cross-entropy loss")
    ax.set_xlim(0, block)
    ax.set_title("Context utilization — later tokens are predicted better\n"
                 "(the model conditions on thousands of preceding tokens)",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ASSETS.mkdir(exist_ok=True)
    save(fig, str(ASSETS / "context_utilization"))
    print(f"saved assets/context_utilization.{{png,pdf}}  early {early:.3f} → late {late:.3f}")


if __name__ == "__main__":
    main()
