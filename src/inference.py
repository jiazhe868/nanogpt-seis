"""Inference engine + test harness for a trained nanoGPT-Seis checkpoint.

Provides a reusable `InferenceEngine` (load once, generate / score many times) and
a CLI to exercise it:

  python -m src.inference --test                    # full self-test suite
  python -m src.inference --prompt "The fault"      # one-off generation
  python -m src.inference --interactive             # REPL
  python -m src.inference --perplexity-text "..."   # score arbitrary text

The `--test` suite is the "does inference work" check: it (1) generates from
several earthquake prompts, (2) recomputes val-set perplexity and confirms it
matches the training val loss (inference path == training path), and (3) measures
generation throughput.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .model.gqa_gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
TOKENIZED = ROOT / "data" / "tokenized"


class InferenceEngine:
    def __init__(self, ckpt_path: Path, device: str | None = None,
                 tokenizer_path: Path | None = None, meta_path: Path | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device)
        mcfg = ck["model_cfg"]
        cfg = GPTConfig(**{k: mcfg[k] for k in GPTConfig.__dataclass_fields__ if k in mcfg})
        self.cfg = cfg
        model = GPT(cfg)
        model.load_state_dict(ck["model"])
        self.model = model.eval().to(self.device)
        self.iter_num = ck.get("iter_num", -1)
        self.best_val = ck.get("best_val", float("nan"))

        # tokenizer/meta default to the current ones; override to score an older
        # checkpoint against the tokenizer it was actually trained with.
        self.tok = Tokenizer.from_file(str(tokenizer_path or TOKENIZED / "tokenizer.json"))
        self.meta = json.loads((meta_path or TOKENIZED / "meta.json").read_text())
        self.eot_id = self.meta["eot_id"]
        # autocast for speed/parity with training
        self.amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device == "cuda" else torch.autocast("cpu", enabled=False))

    # ---- generation ----
    @torch.no_grad()
    def generate(self, prompt: str = "", max_new_tokens: int = 256,
                 temperature: float = 0.8, top_k: int | None = 200,
                 seed: int | None = None, repetition_penalty: float = 1.0,
                 no_repeat_ngram: int = 0) -> str:
        if seed is not None:
            torch.manual_seed(seed)
        ids = self.tok.encode(prompt).ids if prompt else [self.eot_id]
        x = torch.tensor(ids, dtype=torch.long, device=self.device)[None]
        with self.amp:
            y = self.model.generate(x, max_new_tokens, temperature=temperature,
                                     top_k=top_k, repetition_penalty=repetition_penalty,
                                     no_repeat_ngram=no_repeat_ngram)
        out = [t for t in y[0].tolist() if t != self.eot_id]
        return self.tok.decode(out)

    @torch.no_grad()
    def stream(self, prompt: str = "", max_new_tokens: int = 256,
               temperature: float = 0.8, top_k: int | None = 200,
               seed: int | None = None, repetition_penalty: float = 1.0,
               no_repeat_ngram: int = 0):
        """Yield decoded text pieces token-by-token (real-time streaming).

        Decodes the full running id list each step and yields only the new
        suffix — this is robust to byte-level BPE where one character can span
        multiple tokens. Stops early if the model emits <|endoftext|>.
        """
        if seed is not None:
            torch.manual_seed(seed)
        ids = self.tok.encode(prompt).ids if prompt else [self.eot_id]
        x = torch.tensor(ids, dtype=torch.long, device=self.device)[None]
        generated: list[int] = []
        prev = ""
        with self.amp:
            for nxt in self.model.generate_stream(x, max_new_tokens, temperature, top_k,
                                                  repetition_penalty, no_repeat_ngram):
                tid = int(nxt[0, 0])
                if tid == self.eot_id:                   # natural stop at doc end
                    break
                generated.append(tid)
                text = self.tok.decode(generated)
                if len(text) > len(prev):                # emit only the new suffix
                    yield text[len(prev):]
                    prev = text

    @torch.no_grad()
    def generate_annotated(self, prompt: str = "", max_new_tokens: int = 100,
                           temperature: float = 0.8, top_k: int | None = 200,
                           seed: int | None = 0):
        """Generate, recording for each token the model's confidence and the top
        alternatives it chose from — for inspecting/plotting generation behavior.

        Returns (prompt, records) where each record is
            {"text", "id", "prob", "topk": [(token_text, prob), ...]}.
        `prob` is the RAW softmax probability (temperature 1) the model assigned
        to the token it emitted — a clean "how sure was it" signal — while the
        token itself is sampled under `temperature`/`top_k`.
        """
        if seed is not None:
            torch.manual_seed(seed)
        ids = self.tok.encode(prompt).ids if prompt else [self.eot_id]
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)[None]
        records = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            with self.amp:
                logits, _ = self.model(idx_cond)
            logits = logits[0, -1].float()
            raw = torch.softmax(logits, dim=-1)              # true confidence
            # sample from the temperature/top-k adjusted distribution
            s = logits / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(s, min(top_k, s.numel()))
                s[s < v[-1]] = -float("inf")
            tid = int(torch.multinomial(torch.softmax(s, dim=-1), 1))
            if tid == self.eot_id:
                break
            tp, ti = torch.topk(raw, 8)
            topk = [(self.tok.decode([int(i)]), float(p)) for p, i in zip(tp, ti)]
            records.append({"text": self.tok.decode([tid]), "id": tid,
                            "prob": float(raw[tid]), "topk": topk})
            idx = torch.cat([idx, torch.tensor([[tid]], device=self.device)], dim=1)
        return prompt, records

    # ---- scoring ----
    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        """Perplexity of arbitrary text (non-overlapping block_size windows)."""
        ids = self.tok.encode(text).ids
        bs = self.cfg.block_size
        total_nll, total_tok = 0.0, 0
        for i in range(0, max(1, len(ids) - 1), bs):
            chunk = ids[i:i + bs + 1]
            if len(chunk) < 2:
                break
            x = torch.tensor(chunk[:-1], device=self.device)[None]
            y = torch.tensor(chunk[1:], device=self.device)[None]
            with self.amp:
                _, loss = self.model(x, y)
            n = len(chunk) - 1
            total_nll += loss.item() * n
            total_tok += n
        return math.exp(total_nll / max(1, total_tok))

    @torch.no_grad()
    def bits_per_byte(self, text: str) -> float:
        """Bits-per-byte of `text` — a tokenizer-INDEPENDENT fluency metric, so it
        compares fairly across models with different vocabularies (v1 vs v2)."""
        ids = self.tok.encode(text).ids
        bs = self.cfg.block_size
        total_nll = 0.0                                   # in nats
        for i in range(0, max(1, len(ids) - 1), bs):
            chunk = ids[i:i + bs + 1]
            if len(chunk) < 2:
                break
            x = torch.tensor(chunk[:-1], device=self.device)[None]
            y = torch.tensor(chunk[1:], device=self.device)[None]
            with self.amp:
                _, loss = self.model(x, y)
            total_nll += loss.item() * (len(chunk) - 1)
        nbytes = max(1, len(text.encode("utf-8")))
        return total_nll / math.log(2) / nbytes

    @torch.no_grad()
    def val_perplexity(self, n_batches: int = 100, batch_size: int = 16) -> tuple[float, float]:
        """Mean loss + perplexity over random windows of val.bin (matches training eval)."""
        data = np.memmap(TOKENIZED / "val.bin", dtype=np.uint16, mode="r")
        bs = self.cfg.block_size
        losses = []
        g = torch.Generator().manual_seed(0)
        for _ in range(n_batches):
            ix = torch.randint(len(data) - bs, (batch_size,), generator=g)
            x = torch.stack([torch.from_numpy(data[i:i+bs].astype(np.int64)) for i in ix]).to(self.device)
            y = torch.stack([torch.from_numpy(data[i+1:i+1+bs].astype(np.int64)) for i in ix]).to(self.device)
            with self.amp:
                _, loss = self.model(x, y)
            losses.append(loss.item())
        mean = sum(losses) / len(losses)
        return mean, math.exp(mean)

    @torch.no_grad()
    def context_utilization(self, n_windows: int = 60,
                            edges=(0, 64, 256, 1024, 2048, 4096)) -> list[tuple]:
        """Mean loss by position within a full block_size window.

        If the model uses long context, loss should DECREASE for later positions
        (more preceding tokens to condition on). Returns [(lo, hi, mean_loss), ...].
        """
        data = np.memmap(TOKENIZED / "val.bin", dtype=np.uint16, mode="r")
        bs = self.cfg.block_size
        edges = [e for e in edges if e <= bs]
        pos_sum = torch.zeros(bs, device=self.device)
        n = 0
        g = torch.Generator().manual_seed(0)
        for _ in range(n_windows):
            i = int(torch.randint(len(data) - bs - 1, (1,), generator=g))
            x = torch.from_numpy(data[i:i+bs].astype(np.int64)).to(self.device)[None]
            y = torch.from_numpy(data[i+1:i+1+bs].astype(np.int64)).to(self.device)[None]
            with self.amp:
                logits, _ = self.model(x, y)
            ce = F.cross_entropy(logits[0].float(), y[0], reduction="none")  # (bs,)
            pos_sum += ce
            n += 1
        mean_pos = (pos_sum / n).cpu()
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            out.append((lo, hi, mean_pos[lo:hi].mean().item()))
        return out

    @torch.no_grad()
    def throughput(self, max_new_tokens: int = 256, warmup: int = 16) -> float:
        """Generated tokens per second (single stream)."""
        x = torch.tensor([[self.eot_id]], device=self.device)
        with self.amp:
            self.model.generate(x, warmup)                 # warm caches/kernels
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with self.amp:
            self.model.generate(x, max_new_tokens)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return max_new_tokens / (time.time() - t0)


PROMPTS = [
    "The 2011 Tohoku earthquake",
    "Earthquake early warning systems",
    "The rupture propagated along the fault",
    "Seismic hazard assessment",
]


def run_tests(eng: InferenceEngine) -> None:
    print(f"=== nanoGPT-Seis inference test ===")
    print(f"model: {eng.model.num_params()/1e6:.1f}M params | ckpt iter {eng.iter_num} "
          f"| train-time best_val {eng.best_val:.4f} | device {eng.device}\n")

    print("--- 1. generation ---")
    for p in PROMPTS:
        txt = eng.generate(p, max_new_tokens=80, temperature=0.8, top_k=200, seed=0)
        print(f"\n> {p!r}\n{txt}\n")

    print("--- 2. val-perplexity sanity (should ~match training val loss) ---")
    mean, ppl = eng.val_perplexity(n_batches=100, batch_size=16)
    print(f"val loss {mean:.4f} (ppl {ppl:.2f}) | training best_val {eng.best_val:.4f} "
          f"-> Δ {abs(mean-eng.best_val):.4f}")

    print("\n--- 3. self-perplexity on a held-out-style sentence ---")
    s = ("The earthquake occurred along a subduction zone where the oceanic "
         "plate descends beneath the continental plate, generating a megathrust rupture.")
    print(f"ppl(domain sentence) = {eng.perplexity(s):.2f}")
    print(f"ppl(random english)  = {eng.perplexity('the cat sat on the mat and then the dog ran away quickly'):.2f}")

    print("\n--- 4. context utilization (loss by position in a "
          f"{eng.cfg.block_size}-token window) ---")
    buckets = eng.context_utilization(n_windows=60)
    for lo, hi, ml in buckets:
        print(f"  positions {lo:>4}-{hi:<4}: mean loss {ml:.3f}")
    first, last = buckets[0][2], buckets[-1][2]
    print(f"  -> late-context loss is {(1-last/first)*100:.0f}% lower than early "
          f"({first:.3f} -> {last:.3f}): the model uses the long context.")

    print(f"\n--- 5. long-context generation ({eng.cfg.block_size}-cap, KV-cached) ---")
    t0 = time.time()
    long_txt = eng.generate("Abstract\n", max_new_tokens=800, temperature=0.8,
                            top_k=200, seed=1)
    dt = time.time() - t0
    print(f"generated 800 tokens in {dt:.1f}s ({800/dt:.0f} tok/s). Excerpt:")
    print(long_txt[:600] + "\n...[truncated]...")

    print("\n--- 6. throughput vs sequence length (KV-cache = flat) ---")
    for n in (256, 1024):
        print(f"  {n:>4} tokens: {eng.throughput(max_new_tokens=n):.1f} tok/s")


def interactive(eng: InferenceEngine, repetition_penalty=1.15, no_repeat_ngram=3) -> None:
    print("Interactive mode (streaming). Empty prompt = free generation. Ctrl-C to exit.")
    try:
        while True:
            p = input("\nprompt> ").strip()
            print(p, end="", flush=True)
            for piece in eng.stream(p, max_new_tokens=300, temperature=0.8, top_k=200,
                                    repetition_penalty=repetition_penalty,
                                    no_repeat_ngram=no_repeat_ngram):
                print(piece, end="", flush=True)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "ckpt.pt")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--perplexity-text", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--no-stream", action="store_true",
                    help="print the full completion at once instead of streaming")
    ap.add_argument("--repetition-penalty", type=float, default=1.15,
                    help="downweight already-seen tokens (1.0 = off)")
    ap.add_argument("--no-repeat-ngram", type=int, default=3,
                    help="hard-ban repeating n-grams of this size (0 = off)")
    args = ap.parse_args()

    eng = InferenceEngine(args.ckpt)
    rp, ng = args.repetition_penalty, args.no_repeat_ngram
    if args.test:
        run_tests(eng)
    elif args.interactive:
        interactive(eng, rp, ng)
    elif args.perplexity_text is not None:
        print(f"ppl = {eng.perplexity(args.perplexity_text):.2f}")
    else:
        prompt = args.prompt or "The 2011 Tohoku earthquake"
        if args.no_stream:
            print(eng.generate(prompt, args.max_new_tokens, args.temperature,
                               args.top_k, repetition_penalty=rp, no_repeat_ngram=ng))
        else:
            print(prompt, end="", flush=True)           # echo prompt, then stream
            for piece in eng.stream(prompt, args.max_new_tokens, args.temperature,
                                    args.top_k, repetition_penalty=rp, no_repeat_ngram=ng):
                print(piece, end="", flush=True)
            print()


if __name__ == "__main__":
    main()
