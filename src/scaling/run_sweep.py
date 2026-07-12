"""Generate the IsoFLOP config family and (optionally) launch the runs.

Two steps, kept separate so you can inspect before spending GPU time:

  # 1. write configs/scaling/*.yaml + a manifest, and print the run matrix
  python -m src.scaling.run_sweep --generate

  # 2. launch every pending run on 2 GPUs, sequentially, resumable
  python -m src.scaling.run_sweep --run --nproc 2

Each run trains one model under one compute budget C at a fixed data budget
D = C/(6N) (see spec.py). Runs are resumable: a finished run drops a `DONE` marker
in its out_dir and is skipped on re-invocation; an interrupted run continues from its
last checkpoint (`src.train --resume`). Restrict to some budgets with
`--budgets 2e16 6e16`, or preview with `--dry-run`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from .spec import BLOCK, VOCAB, build_matrix

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "scaling"

# Rough combined effective throughput of 2× A30 in bf16 on these small models,
# used only to print a time estimate. Real numbers will vary.
EFF_FLOPS = 2.5e13


def _config_for(run: dict) -> dict:
    m = run["model"]
    return {
        "model": {
            "vocab_size": VOCAB,          # pinned again from data/tokenized at train time
            "block_size": BLOCK,
            "n_layer": m["n_layer"],
            "n_head": m["n_head"],
            "n_kv_head": m["n_kv_head"],
            "d_model": m["d_model"],
            "ffn_multiple_of": 256,
            "rope_theta": 10000.0,
            "dropout": 0.0,
            "mup": run["mup"],
            "mup_base_width": run["mup_base_width"],
        },
        "train": {
            "batch_size": m["batch_size"],
            "grad_accum": run["grad_accum"],
            "block_size": BLOCK,
            "max_iters": run["max_iters"],
            "warmup_iters": run["warmup_iters"],
            "lr": run["lr"],
            "min_lr": run["min_lr"],
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "eval_interval": run["eval_interval"],
            "eval_iters": 50,
            "log_interval": 20,
            "compile": True,
            "dtype": "bfloat16",
            "seed": 1337,
            "out_dir": run["out_dir"],
        },
    }


def _fmt(runs: list[dict]) -> str:
    lines = [f"{'run':>16} {'N(non-emb)':>12} {'D(tokens)':>14} "
             f"{'iters':>7} {'C(FLOPs)':>10} {'~min':>6}  keep"]
    total_flops = 0.0
    for r in runs:
        keep = "yes" if r["in_window"] else "SKIP (D out of window)"
        mins = r["flops"] / EFF_FLOPS / 60 if r["in_window"] else 0
        if r["in_window"]:
            total_flops += r["flops"]
        lines.append(
            f"{r['run_name']:>16} {r['N']:>12,} {r['D']:>14,.0f} "
            f"{r['max_iters']:>7,} {r['budget']:>10.1e} {mins:>6.0f}  {keep}")
    lines.append(f"\ntotal compute (kept runs): {total_flops:.2e} FLOPs "
                 f"≈ {total_flops / EFF_FLOPS / 3600:.1f} h at {EFF_FLOPS:.0e} FLOP/s")
    return "\n".join(lines)


def generate(runs: list[dict]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CONFIG_DIR.glob("*.yaml"):     # drop configs from a previous spec
        stale.unlink()
    manifest = []
    for r in runs:
        if not r["in_window"]:
            continue
        path = CONFIG_DIR / f"{r['run_name']}.yaml"
        path.write_text(yaml.safe_dump(_config_for(r), sort_keys=False))
        manifest.append({
            "run_name": r["run_name"], "config": str(path.relative_to(ROOT)),
            "out_dir": r["out_dir"], "budget": r["budget"], "N": r["N"],
            "D": r["D"], "max_iters": r["max_iters"], "model": r["model"]["name"],
        })
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(_fmt(runs))
    print(f"\nwrote {len(manifest)} configs + manifest.json to {CONFIG_DIR}")


def run(runs: list[dict], nproc: int, dry: bool) -> None:
    pending = [r for r in runs if r["in_window"]]
    print(f"{len(pending)} runs to execute on {nproc} GPU(s)\n")
    for i, r in enumerate(pending, 1):
        out_dir = ROOT / r["out_dir"]
        done = out_dir / "DONE"
        cfg = CONFIG_DIR / f"{r['run_name']}.yaml"
        if done.exists():
            print(f"[{i}/{len(pending)}] {r['run_name']}: DONE, skipping")
            continue
        if not cfg.exists():
            print(f"[{i}/{len(pending)}] {r['run_name']}: config missing — run --generate first")
            continue
        if nproc > 1:
            cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}",
                   "-m", "src.train", "--config", str(cfg), "--resume"]
        else:
            cmd = [sys.executable, "-m", "src.train", "--config", str(cfg), "--resume"]
        print(f"[{i}/{len(pending)}] {r['run_name']}  "
              f"(N={r['N']:,}  D={r['D']:,.0f}  iters={r['max_iters']:,})")
        print("    $", " ".join(cmd))
        if dry:
            continue
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode == 0:
            done.write_text("ok\n")
            print(f"    ✓ done → {r['out_dir']}/DONE")
        else:
            print(f"    ✗ exited {proc.returncode}; not marking DONE (re-run to resume)")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="write configs + manifest")
    parser.add_argument("--run", action="store_true", help="launch pending runs")
    parser.add_argument("--nproc", type=int, default=2, help="GPUs per run (torchrun)")
    parser.add_argument("--budgets", type=float, nargs="*", default=None,
                    help="restrict to these compute budgets (FLOPs)")
    parser.add_argument("--dry-run", action="store_true", help="with --run: print commands only")
    args = parser.parse_args()

    runs = build_matrix(nproc=args.nproc, budgets=args.budgets)
    if not (args.generate or args.run):
        print(_fmt(runs))
        print("\n(nothing launched — pass --generate to write configs, --run to train)")
        return
    if args.generate:
        generate(runs)
    if args.run:
        run(runs, args.nproc, args.dry_run)


if __name__ == "__main__":
    main()
