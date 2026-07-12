"""Model-architecture diagram (graphviz, bottom-to-top).

Input at the bottom, logits at the top. One transformer block is expanded inside
a cluster (it repeats ×16). Pastels categorize function: blue = embeddings,
yellow = attention, green = feed-forward, pink = normalization, neutral = I/O and
residual adds. Solid charcoal edges = main data flow; dashed = residual skips and
weight tying.
"""
from __future__ import annotations

from pathlib import Path

from graphviz import Digraph

from ._style import (GV_BG, GV_BLUE, GV_BORDER, GV_CLUSTER, GV_EDGE, GV_GREEN,
                     GV_NEUTRAL, GV_PINK, GV_YELLOW, gv_render)

ASSETS = Path(__file__).resolve().parents[2] / "assets"
TIE = "#B07A00"          # muted gold for the weight-tying link


def main():
    dot = Digraph("architecture")
    dot.attr(rankdir="BT", bgcolor=GV_BG, dpi="300", nodesep="0.4", ranksep="0.5",
           splines="ortho", fontname="Arial", labelloc="t", fontsize="20",
           label="nanoGPT-Seis — 113M decoder (d_model 768 · 16 layers · "
                 "GQA 12:4 · SwiGLU 2048 · ctx 4096)")
    dot.attr("node", shape="box", style="filled,rounded", fontname="Arial",
           color=GV_BORDER, penwidth="1.0", fontsize="11", margin="0.20,0.09")
    dot.attr("edge", color=GV_EDGE, penwidth="1.2", arrowsize="0.8")

    dot.node("input", "input token ids  (B, T)", fillcolor=GV_NEUTRAL)
    dot.node("embed", "Token Embedding\n16384 × 768", fillcolor=GV_BLUE)

    with dot.subgraph(name="cluster_block") as c:
        c.attr(label="Transformer block  × 16   (pre-norm, residual)",
               style="rounded", color=GV_CLUSTER, fontname="Arial",
               fontsize="11", fontcolor=GV_EDGE)
        c.node("n1", "RMSNorm", fillcolor=GV_PINK)
        c.node("attn", "Grouped-Query Attention\nRoPE on Q,K · Flash SDPA", fillcolor=GV_YELLOW)
        c.node("add1", "⊕  add residual", fillcolor=GV_NEUTRAL)
        c.node("n2", "RMSNorm", fillcolor=GV_PINK)
        c.node("mlp", "SwiGLU MLP\n768 → 2048 → 768", fillcolor=GV_GREEN)
        c.node("add2", "⊕  add residual", fillcolor=GV_NEUTRAL)

    dot.node("fnorm", "Final RMSNorm", fillcolor=GV_PINK)
    dot.node("head", "LM Head\n768 × 16384", fillcolor=GV_BLUE)
    dot.node("logits", "logits  (B, T, 16384)", fillcolor=GV_NEUTRAL)

    # main data flow (bottom → top)
    flow = ["input", "embed", "n1", "attn", "add1", "n2", "mlp", "add2",
            "fnorm", "head", "logits"]
    for a, b in zip(flow[:-1], flow[1:]):
        dot.edge(a, b)

    # residual skips (dashed; don't constrain the layout)
    dot.edge("embed", "add1", style="dashed", constraint="false")
    dot.edge("add1", "add2", style="dashed", constraint="false")

    # weight tying between embedding and LM head (dashed, muted gold).
    # ortho routing ignores edge `label`, so use `xlabel` to place it near the edge.
    dot.edge("embed", "head", style="dashed", constraint="false",
           color=TIE, fontcolor=TIE, fontname="Arial", fontsize="9",
           xlabel="weight tying")

    ASSETS.mkdir(exist_ok=True)
    gv_render(dot, str(ASSETS / "architecture"))
    print("saved assets/architecture.{png,pdf}")


if __name__ == "__main__":
    main()
