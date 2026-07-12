"""The IsoFLOP experiment design: model family, compute budgets, run matrix.

The plan (Chinchilla-style compute-optimal frontier, Hoffmann et al. 2022):

  * Fix the context length and the global batch (tokens/iter) across every run,
    so the only things that vary are the model size N and the data D.
  * For each compute budget C, train every model whose implied data size
    D = C / (6·N) is trainable and under ~4 epochs of the corpus. Each such run
    costs exactly C FLOPs (that is what "IsoFLOP" means), so a budget is a set of
    (N, D) points at constant compute. The N that minimises loss on that curve is
    the compute-optimal size N_opt(C); connecting the minima across budgets gives
    the frontier.

  * N is counted as NON-EMBEDDING parameters (Kaplan et al. convention): with a
    fixed 16k tied embedding the vocab matrix would otherwise dominate tiny models
    and distort the power law. The trainer pins vocab_size from data/tokenized, so
    every run shares the exact same tokenizer and embedding size.

FLOPs use the standard training estimate C ≈ 6·N·D (fwd+bwd), N = non-embedding.
"""
from __future__ import annotations

from functools import lru_cache

VOCAB = 16384
BLOCK = 1024                 # fixed context length for all scaling runs
TOKENS_PER_ITER = 131_072    # 2**17 — fixed global batch (tokens/iter) for every run

# Compute budgets for the IsoFLOP profiles. Chosen so each profile keeps 3+ models
# inside the trainable D-window and the optimum shifts across sizes between budgets.
# A 90x span in C (six 3x steps) gives the frontier fit a long lever arm.
BUDGETS = [6.0e15, 2.0e16, 6.0e16, 1.8e17, 5.4e17]

# Only keep a run if its implied D = C/(6N) is a sane, <~4-epoch amount of data.
D_MIN = 20e6
D_MAX = 3.3e9                # ~4 epochs of the 822.7M-token corpus

# muP LR transfer: tune ONE base LR at the base width and let muP scale the
# hidden-matrix LR by base_width / d_model for every other size (the trainer does
# this per parameter group). So every run shares the same base LR — no per-size
# hand-tuning, and the IsoFLOP fits aren't confounded by an LR that drifts with N.
MUP_BASE_WIDTH = 256      # == the xs model's d_model; LR is "tuned" here
BASE_LR = 1.2e-3          # base learning rate at MUP_BASE_WIDTH

# The model family. head_dim is fixed at 64, so d_model = 64 * n_head (and the muP
# 1/d attention correction is unnecessary — head_dim doesn't change). Width and depth
# scale together. batch_size is picked to fit 24 GB at ctx 1024 with room to spare;
# grad_accum is derived so tokens/iter == TOKENS_PER_ITER.
MODELS = [
    dict(name="xxs", n_layer=3,  n_head=2,  n_kv_head=1, d_model=128, batch_size=32),
    dict(name="xs",  n_layer=4,  n_head=4,  n_kv_head=2, d_model=256, batch_size=32),
    dict(name="s",   n_layer=6,  n_head=6,  n_kv_head=2, d_model=384, batch_size=32),
    dict(name="m",   n_layer=8,  n_head=8,  n_kv_head=2, d_model=512, batch_size=16),
    dict(name="l",   n_layer=10, n_head=10, n_kv_head=2, d_model=640, batch_size=16),
    dict(name="xl",  n_layer=12, n_head=12, n_kv_head=4, d_model=768, batch_size=16),
    dict(name="2xl", n_layer=14, n_head=14, n_kv_head=2, d_model=896, batch_size=8),
]


@lru_cache(maxsize=None)
def non_embedding_params(name: str) -> int:
    """Exact non-embedding parameter count for a model, from the real module."""
    import torch  # local import so config-only tooling need not load torch eagerly

    from ..model.gqa_gpt import GPT, GPTConfig
    m = next(mm for mm in MODELS if mm["name"] == name)
    with torch.no_grad():
        model = GPT(GPTConfig(
            vocab_size=VOCAB, block_size=BLOCK,
            n_layer=m["n_layer"], n_head=m["n_head"], n_kv_head=m["n_kv_head"],
            d_model=m["d_model"], ffn_multiple_of=256, rope_theta=10000.0, dropout=0.0,
        ))
        n = model.num_params(non_embedding=True)
    del model
    return int(n)


def grad_accum_for(batch_size: int, nproc: int) -> int:
    """Micro-steps needed so batch·nproc·block·accum == TOKENS_PER_ITER."""
    denom = batch_size * nproc * BLOCK
    if TOKENS_PER_ITER % denom != 0:
        raise ValueError(
            f"TOKENS_PER_ITER ({TOKENS_PER_ITER}) not divisible by "
            f"batch*nproc*block ({batch_size}*{nproc}*{BLOCK}={denom}); "
            f"pick a batch_size that divides {TOKENS_PER_ITER // (nproc * BLOCK)}.")
    return TOKENS_PER_ITER // denom


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def budget_tag(C: float) -> str:
    """Compact, filename-safe label for a budget: 2e16, 6e16, 1.8e17."""
    mant, exp = f"{C:.1e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


def build_matrix(nproc: int = 2, budgets: list[float] | None = None) -> list[dict]:
    """Resolve every (budget, model) pair into a concrete run spec.

    Returns one dict per run with the model shape, the derived data budget D, the
    training length in iters, and per-run schedule/optimizer settings. Runs whose
    implied D falls outside [D_MIN, D_MAX] are dropped (and reported by the caller).
    """
    budgets = budgets or BUDGETS
    runs: list[dict] = []
    for C in budgets:
        for m in MODELS:
            N = non_embedding_params(m["name"])
            D = C / (6.0 * N)
            in_window = D_MIN <= D <= D_MAX
            accum = grad_accum_for(m["batch_size"], nproc)
            tpi = accum * nproc * m["batch_size"] * BLOCK
            max_iters = round(D / tpi)
            run_name = f"C{budget_tag(C)}_{m['name']}"
            runs.append({
                "run_name": run_name,
                "budget": C,
                "model": m,
                "N": N,
                "D": D,
                "in_window": in_window,
                "grad_accum": accum,
                "tokens_per_iter": tpi,
                "max_iters": max_iters,
                "warmup_iters": _clamp(round(0.02 * max_iters), 50, 400),
                "eval_interval": _clamp(round(max_iters / 20), 25, 500),
                "lr": BASE_LR,                 # same base LR for every size; muP scales it
                "min_lr": BASE_LR / 10.0,
                "mup": True,
                "mup_base_width": MUP_BASE_WIDTH,
                "out_dir": f"checkpoints/scaling/{run_name}",
                "flops": C,   # by construction each run is exactly C FLOPs
            })
    return runs
