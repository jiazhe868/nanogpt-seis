"""What the model learned — a 2-D PCA of token embeddings.

Curated seismology terms vs everyday words; if the model learned domain structure
they separate into clusters. (Nearest-neighbour lists are printed for reference.)
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ._style import BLUE, GRID, INK, ORANGE, apply_base, save
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"

SEIS = ["earthquake", "seismic", "fault", "rupture", "magnitude", "tsunami",
        "epicenter", "aftershock", "slip", "tectonic", "subduction", "fracture",
        "hypocenter", "waveform"]
GEN = ["history", "government", "economy", "music", "water", "city", "science",
       "animal", "language", "market", "weather", "church", "river", "school"]


def main():
    from src.inference import InferenceEngine
    eng = InferenceEngine(ROOT / "checkpoints" / "ckpt.pt")
    E = eng.model.tok_emb.weight.float().detach()
    En = F.normalize(E, dim=1)

    words, vecs, grp = [], [], []
    for w, g in [(w, "seis") for w in SEIS] + [(w, "gen") for w in GEN]:
        ids = eng.tok.encode(" " + w).ids
        if len(ids) != 1:
            continue
        words.append(w)
        vecs.append(E[ids[0]])
        grp.append(g)

    X = torch.stack(vecs)
    X = X - X.mean(0)
    _, _, V = torch.pca_lowrank(X, q=2)
    proj = (X @ V[:, :2]).cpu().numpy()

    apply_base()
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    for (px, py), word, group in zip(proj, words, grp):
        color = ORANGE if group == "seis" else BLUE
        ax.scatter([px], [py], s=45, color=color, zorder=3, edgecolor="white", linewidth=0.6)
        ax.annotate(word, (px, py), fontsize=9.5, color=INK,
                    xytext=(5, 3), textcoords="offset points")
    ax.set_title("Token-embedding space (PCA) — seismology terms cluster apart "
                 "from everyday words", loc="left", fontsize=12)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, label="seismology"),
                       Line2D([], [], marker="o", ls="", color=BLUE, label="everyday")],
              frameon=False, loc="best")
    ASSETS.mkdir(exist_ok=True)
    save(fig, str(ASSETS / "embedding_space"))

    # nearest neighbours (reference)
    for term in [" earthquake", " fault", " tsunami"]:
        ids = eng.tok.encode(term).ids
        if len(ids) == 1:
            sim = En @ En[ids[0]]
            nb = [eng.tok.decode([t]).strip() for t in sim.topk(6).indices.tolist()[1:]]
            print(f"  {term.strip():11s} → {', '.join(nb)}")
    print("saved assets/embedding_space.{png,pdf}")


if __name__ == "__main__":
    main()
