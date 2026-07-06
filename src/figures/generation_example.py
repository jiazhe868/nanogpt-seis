"""Visualize a real inference output: generated text shaded by the model's
per-token confidence, plus the top-k distribution at the least-confident token.

  CUDA_VISIBLE_DEVICES=0 python -m src.figures.generation_example \
      --prompt "The 2011 Tohoku earthquake"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable

from ._style import GRID, INK, INK2, MUTED, ORANGE, PANEL, apply_base
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
CMAP = colormaps["Blues"]


def _vis(s):                       # make whitespace visible in labels
    return s.replace("\n", "↵").replace(" ", "␣")


def _color(prob):                  # low conf → light, high conf → dark blue
    return CMAP(0.18 + 0.75 * prob)


def _textcolor(prob):
    return "white" if prob > 0.55 else INK


def lay_out_text(ax, prompt, records):
    """Place the prompt (neutral) then each generated token (confidence-colored),
    measuring each token's real width so there are no gaps, wrapping at the edge."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    x, y, lh, xmax = 0.006, 0.94, 0.058, 0.992

    def put(txt, face, tcol, edge=None):
        nonlocal x, y
        txt = txt.replace("\n", " ")
        if not txt:
            return
        t = ax.text(x, y, txt, fontsize=11, family="monospace", va="top", ha="left",
                    color=tcol, bbox=dict(boxstyle="round,pad=0.12", facecolor=face,
                                          edgecolor=edge or face, linewidth=0.6))
        bb = t.get_window_extent(rend)
        (x0, _), (x1, _) = inv.transform((bb.x0, 0)), inv.transform((bb.x1, 0))
        w = x1 - x0
        if x + w > xmax and x > 0.006:         # wrap
            t.remove()
            x, y = 0.006, y - lh
            t = ax.text(x, y, txt, fontsize=11, family="monospace", va="top", ha="left",
                        color=tcol, bbox=dict(boxstyle="round,pad=0.12", facecolor=face,
                                              edgecolor=edge or face, linewidth=0.6))
            bb = t.get_window_extent(rend)
            (x0, _), (x1, _) = inv.transform((bb.x0, 0)), inv.transform((bb.x1, 0))
            w = x1 - x0
        x += w + 0.001

    # prompt, word by word, in neutral panel colour
    for wtok in re.findall(r"\S+\s*", prompt + " "):
        put(wtok, PANEL, INK, edge=MUTED)
    # generated tokens, coloured by confidence
    for r in records:
        put(r["text"], _color(r["prob"]), _textcolor(r["prob"]))
    return y, lh                       # last line y + line height, for cropping


def bar_panel(ax, rec, step):
    # top raw candidates, plus the actually-emitted token (so we can show why the
    # text panel shades it light: it was a low-probability sample).
    items = list(rec["topk"][:7])
    chosen_raw = rec["text"]
    if chosen_raw not in [t for t, _ in items]:
        items.append((chosen_raw, rec["prob"]))
    items.sort(key=lambda tp: tp[1], reverse=True)
    toks = [_vis(t) for t, _ in items]
    probs = [p for _, p in items]
    chosen = _vis(chosen_raw)
    top1, p1 = _vis(rec["topk"][0][0]), rec["topk"][0][1]
    y = list(range(len(toks)))[::-1]
    colors = [ORANGE if t == chosen else "#9ec5f4" for t in toks]
    ax.barh(y, probs, color=colors, height=0.72)
    for yi, t, p in zip(y, toks, probs):
        ax.text(-0.012, yi, t, ha="right", va="center", fontsize=9,
                family="monospace", color=(ORANGE if t == chosen else INK))
        ax.text(p + 0.008, yi, f"{p:.2f}", ha="left", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, max(probs) * 1.28); ax.set_ylim(-0.6, len(toks) - 0.4)
    ax.set_yticks([]); ax.set_xlabel("P(token)")
    ax.set_title(f"why that token is light — next-token distribution at step {step}:  "
                 f"the model favored {top1!r} (p={p1:.2f}),\nbut temperature sampling "
                 f"emitted {chosen!r} (p={rec['prob']:.2f}, orange)",
                 loc="left", fontsize=10.5)
    ax.grid(True, axis="x", color=GRID, lw=0.9); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def pick_step(records):
    """Least-confident 'content' token (alphabetic, len>2) in the first 60 steps."""
    cand = [(i, r) for i, r in enumerate(records[:60])
            if re.search(r"[A-Za-z]{3,}", r["text"])]
    return min(cand, key=lambda ir: ir[1]["prob"]) if cand else (0, records[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "ckpt.pt")
    ap.add_argument("--prompt", type=str, default="The 2011 Tohoku earthquake")
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.inference import InferenceEngine
    eng = InferenceEngine(args.ckpt)
    prompt, records = eng.generate_annotated(
        args.prompt, max_new_tokens=args.max_new_tokens, temperature=0.8,
        top_k=200, seed=args.seed)
    mean_conf = sum(r["prob"] for r in records) / len(records)
    step, rec = pick_step(records)

    apply_base()
    fig = plt.figure(figsize=(12.5, 7.4))
    gs = GridSpec(2, 2, height_ratios=[2.0, 1.15], width_ratios=[1, 0.02],
                  hspace=0.28, wspace=0.03)
    ax_txt = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])

    fig.suptitle("nanoGPT-Seis — an inference output, shaded by token confidence",
                 x=0.02, y=0.97, ha="left", fontsize=14, fontweight="bold")
    ax_txt.text(0, 1.045, f'prompt: "{prompt}"   ·   mean confidence '
                f'{mean_conf:.2f}   ·   grey = prompt, blue = generated',
                transform=ax_txt.transAxes, fontsize=9.5, color=INK2)

    y_last, lh = lay_out_text(ax_txt, prompt, records)
    # Shrink the text axis (and its colorbar) to exactly the lines used — this
    # keeps the glyph size/spacing constant while removing empty space — then
    # slide the bar panel up under it. bbox_inches="tight" trims the rest.
    bot = y_last - 0.6 * lh
    frac = max(0.15, 1.0 - bot)
    for a in (ax_txt, ax_cb):
        p = a.get_position()
        a.set_position([p.x0, p.y1 - p.height * frac, p.width, p.height * frac])
    ax_txt.set_ylim(bot, 1.0)
    tp = ax_txt.get_position(); bp = ax_bar.get_position()
    ax_bar.set_position([bp.x0, tp.y0 - 0.10 - bp.height, bp.width, bp.height])

    sm = ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=CMAP)
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("P(token) — model confidence", fontsize=9)
    bar_panel(ax_bar, rec, step)

    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "generation_example.png", dpi=150, bbox_inches="tight")
    print(f"saved assets/generation_example.png  (mean conf {mean_conf:.3f}, "
          f"low-conf step {step}: {rec['text']!r} p={rec['prob']:.2f})")


if __name__ == "__main__":
    main()
