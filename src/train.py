"""DDP pretraining loop for the GQA+RoPE model.

Single GPU:
  python -m src.train --config configs/gpt120m.yaml
Two A30s:
  torchrun --standalone --nproc_per_node=2 -m src.train --config configs/gpt120m.yaml

bf16 autocast, cosine LR + warmup, gradient accumulation (no_sync on all but the
last micro-step), grad clipping, periodic eval + best-checkpoint save + resume,
optional torch.compile, CSV logging.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from .model.gqa_gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
TOKENIZED = ROOT / "data" / "tokenized"


# --------------------------------------------------------------------------- #
def load_config(path: Path, overrides: dict) -> tuple[dict, dict]:
    cfg = yaml.safe_load(path.read_text())
    model_cfg, train_cfg = cfg["model"], cfg["train"]
    for key, value in overrides.items():
        if value is None:
            continue
        (model_cfg if key in model_cfg else train_cfg)[key] = value
    return model_cfg, train_cfg


def get_batch(data, block_size, batch_size, device, device_type):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / max(1, (max_iters - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gpt120m.yaml")
    parser.add_argument("--resume", action="store_true")
    # common overrides for quick runs / smoke tests
    parser.add_argument("--max-iters", type=int, default=None, dest="max_iters")
    parser.add_argument("--eval-interval", type=int, default=None, dest="eval_interval")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--grad-accum", type=int, default=None, dest="grad_accum")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    overrides = {
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "compile": False if args.no_compile else None,
    }
    model_cfg, train_cfg = load_config(args.config, overrides)

    # ---- DDP setup ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master = rank == 0
        seed_offset = rank
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        master = True
        seed_offset = 0

    device_type = "cuda" if "cuda" in device else "cpu"
    torch.manual_seed(train_cfg["seed"] + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[train_cfg["dtype"]]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.autocast(device_type=device_type, dtype=dtype))

    # ---- data ----
    meta_path = TOKENIZED / "meta.json"
    if meta_path.exists():
        model_cfg["vocab_size"] = json.loads(meta_path.read_text())["vocab_size"]
    block_size = model_cfg["block_size"]
    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg["grad_accum"]
    train_data = np.memmap(TOKENIZED / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(TOKENIZED / "val.bin", dtype=np.uint16, mode="r")

    tokens_per_iter = grad_accum * world_size * batch_size * block_size
    if master:
        print(f"[train] tokens/iter = {tokens_per_iter:,} "
              f"| train tokens = {len(train_data):,} | vocab = {model_cfg['vocab_size']}")

    # ---- model ----
    gpt_cfg = GPTConfig(**{k: model_cfg[k] for k in GPTConfig.__dataclass_fields__ if k in model_cfg})
    model = GPT(gpt_cfg).to(device)
    raw_model = model
    iter_num, best_val = 0, float("inf")

    out_dir = ROOT / train_cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ck["model"])
        iter_num, best_val = ck["iter_num"], ck["best_val"]
        if master:
            print(f"[train] resumed from iter {iter_num} (best_val {best_val:.4f})")

    optimizer = raw_model.configure_optimizers(
        train_cfg["weight_decay"], train_cfg["lr"], (train_cfg["beta1"], train_cfg["beta2"]), device_type)
    if args.resume and ckpt_path.exists():
        optimizer.load_state_dict(ck["optimizer"])

    if train_cfg["compile"]:
        if master:
            print("[train] compiling model (first step is slow) ...")
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, data in [("train", train_data), ("val", val_data)]:
            losses = torch.zeros(train_cfg["eval_iters"])
            for k in range(train_cfg["eval_iters"]):
                x, y = get_batch(data, block_size, batch_size, device, device_type)
                with ctx:
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    csv_path = out_dir / "log.csv"
    if master and not (args.resume and csv_path.exists()):
        csv_path.write_text("iter,train_loss,val_loss,lr,ms_per_iter\n")

    # ---- training loop ----
    x, y = get_batch(train_data, block_size, batch_size, device, device_type)
    t0 = time.time()
    max_iters = train_cfg["max_iters"]
    while iter_num <= max_iters:
        lr = get_lr(iter_num, train_cfg["warmup_iters"], max_iters, train_cfg["lr"], train_cfg["min_lr"])
        for g in optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)   # lr_mult != 1 only under muP

        if iter_num % train_cfg["eval_interval"] == 0:
            losses = estimate_loss()
            if master:
                dt = (time.time() - t0) * 1000
                print(f"[eval] iter {iter_num}: train {losses['train']:.4f} "
                      f"val {losses['val']:.4f} lr {lr:.2e}")
                with csv_path.open("a") as f:
                    f.write(f"{iter_num},{losses['train']:.4f},{losses['val']:.4f},{lr:.3e},{dt:.0f}\n")
                if losses["val"] < best_val or iter_num == 0:
                    best_val = min(best_val, losses["val"])
                    if iter_num > 0:
                        torch.save({
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iter_num": iter_num, "best_val": best_val,
                            "model_cfg": model_cfg,
                        }, ckpt_path)

        # forward/backward with gradient accumulation
        for micro in range(grad_accum):
            if ddp:
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            x, y = get_batch(train_data, block_size, batch_size, device, device_type)
            loss.backward()
        if train_cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % train_cfg["log_interval"] == 0 and master and iter_num > 0:
            dt = (time.time() - t0) / train_cfg["log_interval"]
            print(f"[train] iter {iter_num}: loss {loss.item()*grad_accum:.4f} "
                  f"{dt*1000:.0f} ms/iter")
            t0 = time.time()
        iter_num += 1

    if ddp:
        destroy_process_group()
    if master:
        print(f"[train] done. best_val={best_val:.4f}  ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
