"""Fit the compute-optimal frontier from finished IsoFLOP runs.

For each compute budget C it fits a parabola to (log N, loss) — the IsoFLOP profile
— and takes the vertex as the compute-optimal size N_opt(C) and loss L_opt(C). It then
fits power laws N_opt ∝ C^a and D_opt ∝ C^b across budgets (Chinchilla predicts
a ≈ b ≈ 0.5). Reads each run's min val loss from its log.csv.

  python -m src.scaling.fit          # after (some) runs finish

Writes checkpoints/scaling/scaling_fit.json, consumed by
src.figures.scaling_laws for the plot.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "scaling" / "manifest.json"
OUT = ROOT / "checkpoints" / "scaling" / "scaling_fit.json"
REF_CONFIG = ROOT / "configs" / "gpt120m_ctx4k.yaml"   # the real 113M run
REF_GPUS = 2                                            # it trained on both A30s


def reference_point() -> dict | None:
    """The real 113M / 823M-token model as a (C, N, D) point, for the frontier plot.

    N is non-embedding params (as in the sweep); D is tokens processed during its run
    (max_iters × tokens/iter). It trained at ctx 4096 on the general+domain mix, so its
    loss is not directly comparable to the ctx-1024 sweep — only its position in
    (C, N) space is.
    """
    if not REF_CONFIG.exists():
        return None
    import yaml
    import torch
    from ..model.gqa_gpt import GPT, GPTConfig
    cfg = yaml.safe_load(REF_CONFIG.read_text())
    m, t = cfg["model"], cfg["train"]
    gc = GPTConfig(**{k: m[k] for k in GPTConfig.__dataclass_fields__ if k in m})
    with torch.no_grad():
        N = GPT(gc).num_params(non_embedding=True)
    D = t["max_iters"] * t["grad_accum"] * REF_GPUS * t["batch_size"] * m["block_size"]
    return {"name": "113M v2 (ctx 4096)", "N": int(N), "D": int(D), "C": 6.0 * N * D}


def _min_val_loss(log_csv: Path) -> float | None:
    if not log_csv.exists():
        return None
    best = math.inf
    with log_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                best = min(best, float(row["val_loss"]))
            except (KeyError, ValueError):
                continue
    return best if math.isfinite(best) else None


def collect() -> list[dict]:
    """Attach each run's measured min val loss to its manifest entry."""
    manifest = json.loads(MANIFEST.read_text())
    runs = []
    for r in manifest:
        loss = _min_val_loss(ROOT / r["out_dir"] / "log.csv")
        if loss is None:
            print(f"  (no log yet: {r['run_name']})")
            continue
        runs.append({**r, "loss": loss})
    return runs


def fit_isoflop(runs: list[dict]) -> list[dict]:
    """One parabola per budget; vertex = compute-optimal (N_opt, D_opt, L_opt)."""
    by_budget: dict[float, list[dict]] = defaultdict(list)
    for r in runs:
        by_budget[r["budget"]].append(r)

    frontier = []
    for C, pts in sorted(by_budget.items()):
        pts = sorted(pts, key=lambda p: p["N"])
        logN = np.array([math.log(p["N"]) for p in pts])
        loss = np.array([p["loss"] for p in pts])
        if len(pts) >= 3:
            a, b, c = np.polyfit(logN, loss, 2)
            if a > 0:                                   # convex: real minimum
                logN_opt = -b / (2 * a)
                L_opt = a * logN_opt**2 + b * logN_opt + c
                method = "parabola"
            else:                                       # not convex: fall back
                i = int(loss.argmin()); logN_opt, L_opt, method = logN[i], loss[i], "argmin"
        else:
            i = int(loss.argmin()); logN_opt, L_opt, method = logN[i], loss[i], "argmin"
        N_opt = math.exp(logN_opt)
        # keep the optimum inside the sampled range; extrapolation is unreliable
        N_opt = min(max(N_opt, math.exp(logN.min())), math.exp(logN.max()))
        frontier.append({
            "budget": C, "N_opt": N_opt, "D_opt": C / (6 * N_opt),
            "L_opt": float(L_opt), "method": method, "n_points": len(pts),
        })
    return frontier


def power_law(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Fit y = k · x^p in log space; return (exponent p, coefficient k)."""
    lx, ly = np.log(xs), np.log(ys)
    p, logk = np.polyfit(lx, ly, 1)
    return float(p), float(math.exp(logk))


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("no manifest — run `python -m src.scaling.run_sweep --generate` first")
    runs = collect()
    if not runs:
        raise SystemExit("no finished runs found (no log.csv with a val_loss yet)")

    frontier = fit_isoflop(runs)
    result = {"runs": runs, "frontier": frontier}

    print(f"\n{'budget C':>10} {'N_opt':>12} {'D_opt':>14} {'L_opt':>7} {'pts':>4} {'fit':>9}")
    for f in frontier:
        print(f"{f['budget']:>10.0e} {f['N_opt']:>12,.0f} {f['D_opt']:>14,.0f} "
              f"{f['L_opt']:>7.3f} {f['n_points']:>4} {f['method']:>9}")

    if len(frontier) >= 2:
        Cs = [f["budget"] for f in frontier]
        aN, kN = power_law(Cs, [f["N_opt"] for f in frontier])
        aD, kD = power_law(Cs, [f["D_opt"] for f in frontier])
        result["exponents"] = {"N_vs_C": aN, "D_vs_C": aD, "kN": kN, "kD": kD}
        print(f"\ncompute-optimal frontier (fitted across {len(frontier)} budgets):")
        print(f"  N_opt ∝ C^{aN:.3f}   D_opt ∝ C^{aD:.3f}   "
              f"(Chinchilla ≈ 0.5 / 0.5)")
        for f in frontier:
            print(f"    C={f['budget']:.1e}: {f['D_opt'] / f['N_opt']:.1f} tokens/param")

        ref = reference_point()
        if ref:
            N_opt_ref = kN * ref["C"] ** aN            # extrapolate the frontier to C_ref
            ref["N_opt_at_C"] = N_opt_ref
            ref["ratio_N_over_Nopt"] = ref["N"] / N_opt_ref
            result["reference"] = ref
            over = "larger than" if ref["ratio_N_over_Nopt"] > 1 else "smaller than"
            print(f"\nreal model {ref['name']}: N={ref['N']:,}  D={ref['D']:,}  "
                  f"C={ref['C']:.2e}")
            print(f"  frontier predicts N_opt(C)={N_opt_ref:,.0f} → the real model is "
                  f"{ref['ratio_N_over_Nopt']:.2f}× that ({over} compute-optimal); "
                  f"{ref['D'] / ref['N']:.1f} tokens/param")
    else:
        print("\n(need ≥2 finished budgets to fit the frontier exponents)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
