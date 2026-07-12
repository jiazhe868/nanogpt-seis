"""A small Llama-style decoder-only Transformer.

The pieces, and why each is here:
  * RMSNorm (pre-norm)  — normalize each token before the sublayer; cheaper and
                          as effective as LayerNorm.
  * RoPE                — inject position by rotating Q/K, so attention scores
                          depend on the *relative* distance between tokens.
  * Grouped-Query Attention (GQA) — many query heads share a few key/value
                          heads, shrinking the KV cache for cheap long-context
                          inference. Runs through F.scaled_dot_product_attention
                          (the fused FlashAttention kernel).
  * SwiGLU feed-forward — a gated MLP that beats a plain GELU MLP per parameter.
  * Weight-tied embedding / LM head — the input and output token matrices are the
                          same tensor, saving parameters and regularizing.

The Q/K/V projections are written with einsum so the per-head structure is
explicit in the tensor contraction rather than hidden inside nn.Linear.

Everything is config-driven; `model.num_params()` reports the exact count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# The per-block matrix weights whose fan-in scales with d_model — the "hidden"
# weights in muP terms. Used for both muP init scaling and the per-group LR rule.
_HIDDEN_SUFFIXES = (".wq.weight", ".wk.weight", ".wv.weight", ".wo.weight",
                    ".w1.weight", ".w2.weight", ".w3.weight")


@dataclass
class GPTConfig:
    vocab_size: int = 16384
    block_size: int = 1024      # maximum context length the model is trained for
    n_layer: int = 16
    n_head: int = 12            # number of query heads
    n_kv_head: int = 4          # number of key/value heads (GQA); must divide n_head
    d_model: int = 768          # residual-stream / embedding width
    ffn_multiple_of: int = 256  # SwiGLU hidden dim is rounded up to a multiple of this
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # Maximal-Update Parametrization (muP): when True, hidden-matrix init and LR are
    # scaled by the width ratio to the base model, so a single base LR transfers
    # across widths. Off by default → identical to standard parametrization.
    mup: bool = False
    mup_base_width: int = 256   # d_model of the model the base LR was tuned on

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    def __post_init__(self):
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square normalization.

    Like LayerNorm but without subtracting the mean or a bias term: just scale
    each vector to unit root-mean-square, then apply a learned per-channel gain.
    The reduction is done in fp32 so bf16/fp16 training stays numerically stable.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


# --------------------------------------------------------------------------- #
# Rotary position embeddings (RoPE)
# --------------------------------------------------------------------------- #
def _rope_tables(head_dim: int, seq_len: int, theta: float):
    """Precompute the cos/sin rotation factors for every position, once.

    Each pair of channels is rotated by an angle that grows with the token's
    position; low-index channels rotate fast, high-index channels slow (the
    `inv_freq` geometric schedule). Returned shape is (1, 1, seq_len, head_dim)
    so it broadcasts over batch and heads.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)                 # angle per (position, channel-pair)
    emb = torch.cat((freqs, freqs), dim=-1)          # duplicate so it lines up with rotate_half
    return emb.cos()[None, None], emb.sin()[None, None]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (x1, x2) -> (-x2, x1): a 90° rotation of each channel pair."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    """Rotate each head vector by its position's angle (the RoPE identity
    x·cos + rotate_half(x)·sin). Applied to Q and K only — after this, the dot
    product q_m · k_n depends on the relative offset (m - n)."""
    return x * cos + _rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class GroupedQueryAttention(nn.Module):
    """Causal self-attention with n_head query heads but only n_kv_head K/V heads.

    Setting n_kv_head == n_head recovers ordinary multi-head attention; setting
    it to 1 gives multi-query attention. In between (we use 12 query : 4 KV) the
    KV cache is n_head/n_kv_head times smaller at almost no quality cost, because
    quality is driven mainly by the number of *query* heads.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.hd = cfg.head_dim
        self.dropout = cfg.dropout
        # Projections stay as nn.Linear (bias-free) so the weights are ordinary
        # (out, in) matrices; we call them via einsum in forward().
        self.wq = nn.Linear(cfg.d_model, self.n_head * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(self.n_head * self.hd, cfg.d_model, bias=False)

    def _project(self, x, weight, n_heads):
        """Project x (B, T, d_model) into per-head vectors (B, n_heads, T, head_dim).

        Viewing the (n_heads*head_dim, d_model) weight as (n_heads, head_dim,
        d_model) makes the head split explicit, and the einsum then reads as
        exactly what attention wants: for every head h and position t, contract
        the input over d_model to get a head_dim vector.
            out[b, h, t, i] = sum_d  x[b, t, d] * W[h, i, d]
        This is numerically identical to `linear(x).view(...).transpose(1, 2)`.
        """
        d = x.size(-1)
        w = weight.view(n_heads, self.hd, d)
        return torch.einsum("btd,hid->bhti", x, w)

    def forward(self, x, cos, sin, kv_cache=None):
        """Args:
            cos/sin : RoPE factors already sliced to this chunk's positions.
            kv_cache: optional (past_k, past_v), each (B, n_kv, T_past, head_dim),
                      for incremental decoding.
        Returns (attention_output, updated_kv_cache).
        """
        q = self._project(x, self.wq.weight, self.n_head)   # (B, n_head, T, hd)
        k = self._project(x, self.wk.weight, self.n_kv)     # (B, n_kv,  T, hd)
        v = self._project(x, self.wv.weight, self.n_kv)

        # Position information enters here, on Q and K only.
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Incremental decoding: append this step's K/V to the running cache. We
        # cache only n_kv heads — that shrunken cache is the whole point of GQA.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_cache = (k, v)

        # SDPA needs matching head counts, so broadcast each KV head to the group
        # of query heads that share it. (This expansion is compute-only; the
        # stored cache above stays small.)
        k_rep, v_rep = k, v
        if self.n_kv != self.n_head:
            rep = self.n_head // self.n_kv
            k_rep = k.repeat_interleave(rep, dim=1)
            v_rep = v.repeat_interleave(rep, dim=1)

        # Prefill processes many positions at once and must be causally masked
        # (q_len == k_len). Single-step decode has one query attending to the
        # whole cache, so no mask is needed (q_len == 1 < k_len).
        is_causal = q.shape[2] == k_rep.shape[2]
        y = F.scaled_dot_product_attention(
            q, k_rep, v_rep, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Concatenate heads back into the residual width and mix them with W_o.
        B, _, T, _ = y.shape
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.hd)
        return self.wo(y), new_cache


# --------------------------------------------------------------------------- #
# Feed-forward
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    """Gated feed-forward: down( silu(gate(x)) * up(x) ).

    The extra `gate` branch (vs a plain two-matrix MLP) lets the network
    modulate each hidden unit multiplicatively — more expressive per parameter.
    Hidden width follows Llama's 8/3·d rule so the 3-matrix SwiGLU has a similar
    parameter count to a 4·d two-matrix MLP.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        multiple = cfg.ffn_multiple_of
        hidden = multiple * ((hidden + multiple - 1) // multiple)   # round up to a multiple
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """One pre-norm transformer block: normalize, apply the sublayer, add back.

    Writing it as `x = x + sublayer(norm(x))` keeps a clean residual highway that
    gradients flow through directly, which is what makes deep stacks trainable.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache


# --------------------------------------------------------------------------- #
# The full model
# --------------------------------------------------------------------------- #
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: the same matrix embeds input tokens and scores output
        # tokens. One tensor, so it is only counted/stored once.
        self.tok_emb.weight = self.lm_head.weight

        # RoPE tables are derived, not learned, so they are non-persistent buffers
        # (recomputed on construction rather than saved in the checkpoint).
        cos, sin = _rope_tables(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down the two projections that write into the residual stream
        # (attention W_o and MLP W_2) by 1/sqrt(2·n_layer). Without this the
        # residual variance grows with depth; this keeps it roughly constant.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        # muP (Yang et al. 2022): relative to the base width, shrink hidden-matrix
        # init by 1/sqrt(width_mult) and the output logits by 1/width_mult, so
        # activations and logits stay width-stable and a base LR (see
        # configure_optimizers) transfers across widths. head_dim is fixed across
        # our width sweep, so the 1/sqrt(head_dim) attention scale needs no change.
        # base width == this width → ratio 1 → exact no-op. Kept as a scalar buffer
        # (not a Python float) so it survives torch.compile / device moves.
        ratio = cfg.mup_base_width / cfg.d_model if cfg.mup else 1.0
        self.register_buffer("output_mult", torch.tensor(ratio, dtype=torch.float32),
                             persistent=False)
        if cfg.mup:
            with torch.no_grad():
                for name, p in self.named_parameters():
                    if name.endswith(_HIDDEN_SUFFIXES):
                        p.mul_(math.sqrt(ratio))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()   # subtract the tied embedding/head matrix
        return n

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. With targets, returns (logits, loss) for
        training; without, returns (last-position logits, None) for generation."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq len {T} > block {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:, :, :T]
        sin = self.rope_sin[:, :, :T]
        for blk in self.blocks:
            x, _ = blk(x, cos, sin)
        x = self.norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            if self.cfg.mup:                       # width-stabilising output multiplier
                logits = logits * self.output_mult.to(logits.dtype)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        # At inference we only need the next-token distribution, so score just the
        # final position — avoids a (B, T, vocab) matmul over the whole sequence.
        logits = self.lm_head(x[:, [-1], :])
        if self.cfg.mup:
            logits = logits * self.output_mult.to(logits.dtype)
        return logits, None

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        """AdamW with decoupled weight decay on matmul/embedding weights (dim >= 2)
        but not 1-D params (RMSNorm gains). Under muP (cfg.mup), hidden-matrix
        weights additionally get a per-group learning-rate multiplier of
        base_width / d_model, so one base LR transfers across the width sweep. The
        training loop reads each group's ``lr_mult`` when it sets the scheduled LR.
        """
        mult = self.cfg.d_model / self.cfg.mup_base_width if self.cfg.mup else 1.0
        buckets: dict[tuple[float, float], list] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_hidden = name.endswith(_HIDDEN_SUFFIXES)
            lr_mult = (1.0 / mult) if (self.cfg.mup and is_hidden) else 1.0
            wd = weight_decay if p.dim() >= 2 else 0.0
            buckets.setdefault((lr_mult, wd), []).append(p)
        groups = [{"params": ps, "weight_decay": wd, "lr_mult": lm}
                  for (lm, wd), ps in buckets.items()]
        return torch.optim.AdamW(groups, lr=lr, betas=betas,
                                 fused=(device_type == "cuda"))

    # ---- generation ---- #
    def _sample(self, logits, temperature, top_k, seq=None,
                repetition_penalty=1.0, no_repeat_ngram=0):
        """Turn last-position logits into one sampled token.

        Order matters: discourage repetition, then top-k truncate, then sample.
          * repetition_penalty (>1): divide the logits of already-seen tokens,
            making the model less likely to loop (CTRL-style).
          * no_repeat_ngram (n>0): hard-ban any token that would complete an
            n-gram already present in `seq` — decisively kills exact loops.
        `seq` is the running (B, T) id sequence; both features are no-ops without it.
        """
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if seq is not None and repetition_penalty != 1.0:
            for b in range(logits.size(0)):
                seen_ids = torch.unique(seq[b])
                seen_logits = logits[b, seen_ids]
                logits[b, seen_ids] = torch.where(
                    seen_logits > 0, seen_logits / repetition_penalty,
                    seen_logits * repetition_penalty)
        if seq is not None and no_repeat_ngram > 1 and seq.size(1) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            for b in range(logits.size(0)):
                ids = seq[b].tolist()
                prefix = tuple(ids[-(n - 1):])
                for i in range(len(ids) - n + 1):
                    if tuple(ids[i:i + n - 1]) == prefix:
                        logits[b, ids[i + n - 1]] = -float("inf")
        if top_k is not None:
            kth_largest, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < kth_largest[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                        repetition_penalty=1.0, no_repeat_ngram=0):
        """Yield one new token (B, 1) at a time, reusing a per-layer KV cache so
        each step costs O(current length) instead of recomputing the whole
        sequence. Total length is capped at block_size; a longer prompt is
        truncated to its last block_size tokens.

        The first iteration ("prefill") runs the whole prompt to fill the cache;
        every later iteration feeds only the single new token.
        """
        max_len = self.cfg.block_size
        idx = idx[:, -max_len:]
        full_seq = idx                             # running sequence, for anti-repeat
        caches = [None] * len(self.blocks)
        pos = 0
        step_input = idx
        for _ in range(max_new_tokens):
            T = step_input.shape[1]
            if pos + T > max_len:                  # reached the context limit
                return
            x = self.drop(self.tok_emb(step_input))
            cos = self.rope_cos[:, :, pos:pos + T]   # RoPE at absolute positions
            sin = self.rope_sin[:, :, pos:pos + T]
            for i, blk in enumerate(self.blocks):
                x, caches[i] = blk(x, cos, sin, caches[i])
            x = self.norm(x)
            logits = self.lm_head(x[:, [-1], :])
            pos += T
            next_tok = self._sample(logits, temperature, top_k, seq=full_seq,
                                    repetition_penalty=repetition_penalty,
                                    no_repeat_ngram=no_repeat_ngram)
            yield next_tok
            full_seq = torch.cat((full_seq, next_tok), dim=1)
            step_input = next_tok                  # next step sees only the new token

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 repetition_penalty=1.0, no_repeat_ngram=0):
        """Convenience wrapper: run generate_stream to completion and return the
        full (prompt + generated) id sequence."""
        idx = idx[:, -self.cfg.block_size:]
        for next_tok in self.generate_stream(idx, max_new_tokens, temperature, top_k,
                                             repetition_penalty, no_repeat_ngram):
            idx = torch.cat((idx, next_tok), dim=1)
        return idx

    @torch.no_grad()
    def generate_naive(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                       repetition_penalty=1.0, no_repeat_ngram=0):
        """Cache-free reference path: re-encodes the whole context each step, so
        it is O(T^2) but supports unbounded sliding-window generation past
        block_size. Kept mainly to validate generate()'s KV cache against it."""
        for _ in range(max_new_tokens):
            idx_window = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_window)
            next_tok = self._sample(logits, temperature, top_k, seq=idx,
                                    repetition_penalty=repetition_penalty,
                                    no_repeat_ngram=no_repeat_ngram)
            idx = torch.cat((idx, next_tok), dim=1)
        return idx
