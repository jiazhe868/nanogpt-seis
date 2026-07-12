"""Training-pipeline diagram (graphviz, left-to-right).

Clean auto-routed nodes/edges: rounded filled boxes, Arial, soft neutral
background, thin dark-gray edges. Pastel fills categorize the stages.
"""
from __future__ import annotations

from pathlib import Path

from graphviz import Digraph

from ._style import (GV_BG, GV_BLUE, GV_BORDER, GV_CLUSTER, GV_EDGE, GV_GREEN,
                     GV_PINK, GV_PURPLE, GV_TEAL, GV_YELLOW, gv_render)

ASSETS = Path(__file__).resolve().parents[2] / "assets"


def main():
    dot = Digraph("workflow")
    dot.attr(rankdir="LR", bgcolor=GV_BG, dpi="300", nodesep="0.4", ranksep="0.8",
           splines="ortho", compound="true", fontname="Arial", labelloc="t",
           fontsize="20", label="nanoGPT-Seis — end-to-end pretraining pipeline")
    dot.attr("node", shape="box", style="filled,rounded", fontname="Arial",
           color=GV_BORDER, penwidth="1.0", fontsize="11", margin="0.18,0.10")
    dot.attr("edge", color=GV_EDGE, penwidth="1.2", arrowsize="0.8")

    # data sources cluster
    with dot.subgraph(name="cluster_src") as c:
        c.attr(label="free data sources  ·  earthquake domain + general (fluency)",
               style="rounded", color=GV_CLUSTER, fontname="Arial",
               fontsize="11", fontcolor=GV_EDGE)
        for nid, label, fill in [
            ("arxiv", "arXiv", GV_BLUE), ("crossref", "Crossref +\nUnpaywall", GV_BLUE),
            ("eartharxiv", "EarthArXiv", GV_BLUE), ("substack", "Substack", GV_BLUE),
            ("wiki", "Wikipedia", GV_TEAL), ("fineweb", "FineWeb-Edu", GV_TEAL),
        ]:
            c.node(nid, label, fillcolor=fill)

    # pipeline stages (each a distinct pastel)
    stages = [
        ("crawl", "1. Crawl\nconcurrent · resumable\nPDF → text + validate", GV_BLUE),
        ("process", "2. Process\nclean · filter\nMinHash dedup · split", GV_GREEN),
        ("tokenize", "3. Tokenize\nbyte-level BPE 16k\nencode → uint16", GV_YELLOW),
        ("model", "4. Model\n113M GQA + RoPE\nRMSNorm · SwiGLU", GV_PURPLE),
        ("train", "5. Train\n2×A30 DDP · bf16\ncosine LR · compile", GV_PINK),
        ("infer", "6. Inference\nKV-cache streaming\nperplexity · figures", GV_TEAL),
    ]
    for nid, label, fill in stages:
        dot.node(nid, label, fillcolor=fill)

    # one clean arrow from the sources cluster into Crawl (ltail clips it to the
    # cluster border) — avoids the fan-in kinks of six separate ortho edges.
    dot.edge("eartharxiv", "crawl", ltail="cluster_src")
    for a, b in zip([s[0] for s in stages][:-1], [s[0] for s in stages][1:]):
        dot.edge(a, b)

    ASSETS.mkdir(exist_ok=True)
    gv_render(dot, str(ASSETS / "workflow"))
    print("saved assets/workflow.{png,pdf}")


if __name__ == "__main__":
    main()
