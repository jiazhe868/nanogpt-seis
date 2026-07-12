"""Attention maps — where a few heads look on a seismology sentence.

F.scaled_dot_product_attention hides the weights, so we recompute the softmax
attention explicitly for one layer (same math as the model's forward).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import colormaps

from ._style import INK, apply_base, save
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
CMAP = colormaps["Blues"]
SENT = "The earthquake ruptured the fault and generated a large tsunami."


def attention_at_layer(eng, ids, layer):
    from src.model.gqa_gpt import _apply_rope
    model = eng.model
    x = torch.tensor(ids, device=eng.device)[None]
    T = len(ids)
    with torch.no_grad():
        hidden = model.drop(model.tok_emb(x))
        cos, sin = model.rope_cos[:, :, :T], model.rope_sin[:, :, :T]
        for i, blk in enumerate(model.blocks):
            if i == layer:
                attn = blk.attn
                x_norm = blk.attn_norm(hidden)
                q = _apply_rope(attn._project(x_norm, attn.wq.weight, attn.n_head), cos, sin)
                k = _apply_rope(attn._project(x_norm, attn.wk.weight, attn.n_kv), cos, sin)
                k = k.repeat_interleave(attn.n_head // attn.n_kv, dim=1)
                scores = (q @ k.transpose(-2, -1)) / (attn.hd ** 0.5)
                scores = scores.masked_fill(
                    torch.triu(torch.ones(T, T, device=eng.device), 1).bool(), float("-inf"))
                return F.softmax(scores, dim=-1)[0].cpu()      # (n_head, T, T)
            hidden, _ = blk(hidden, cos, sin)


def main(layer=8, heads=(0, 3, 6, 9)):
    from src.inference import InferenceEngine
    eng = InferenceEngine(ROOT / "checkpoints" / "ckpt.pt")
    ids = eng.tok.encode(SENT).ids
    toks = [eng.tok.decode([t]).strip() or "·" for t in ids]
    att = attention_at_layer(eng, ids, layer)

    apply_base()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.2))
    for ax, hh in zip(axes.flat, heads):
        ax.imshow(att[hh], cmap=CMAP, vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(len(toks))); ax.set_yticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=90, fontsize=7.5)
        ax.set_yticklabels(toks, fontsize=7.5)
        ax.set_title(f"layer {layer} · head {hh}", fontsize=10, color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Attention maps — each row (query) attends over past tokens (keys); "
                 "causal → lower-triangular", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ASSETS.mkdir(exist_ok=True)
    save(fig, str(ASSETS / "attention_map"))
    print("saved assets/attention_map.{png,pdf}")


if __name__ == "__main__":
    main()
