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
            pk, pv = kv_cache
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        new_cache = (k, v)

        # SDPA needs matching head counts, so broadcast each KV head to the group
        # of query heads that share it. (This expansion is compute-only; the
        # stored cache above stays small.)
        kk, vv = k, v
        if self.n_kv != self.n_head:
            rep = self.n_head // self.n_kv
            kk = k.repeat_interleave(rep, dim=1)
            vv = v.repeat_interleave(rep, dim=1)

        # Prefill processes many positions at once and must be causally masked
        # (q_len == k_len). Single-step decode has one query attending to the
        # whole cache, so no mask is needed (q_len == 1 < k_len).
        is_causal = q.shape[2] == kk.shape[2]
        y = F.scaled_dot_product_attention(
            q, kk, vv, is_causal=is_causal,
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
        m = cfg.ffn_multiple_of
        hidden = m * ((hidden + m - 1) // m)             # round up to a multiple of m
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
        a, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + a
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
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        # At inference we only need the next-token distribution, so score just the
        # final position — avoids a (B, T, vocab) matmul over the whole sequence.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        """AdamW with decoupled weight decay applied to matmul/embedding weights
        (dim >= 2) but not to 1-D params (RMSNorm gains), which shouldn't decay."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
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
                ids = torch.unique(seq[b])
                lv = logits[b, ids]
                logits[b, ids] = torch.where(lv > 0, lv / repetition_penalty,
                                             lv * repetition_penalty)
        if seq is not None and no_repeat_ngram > 1 and seq.size(1) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            for b in range(logits.size(0)):
                s = seq[b].tolist()
                prefix = tuple(s[-(n - 1):])
                for i in range(len(s) - n + 1):
                    if tuple(s[i:i + n - 1]) == prefix:
                        logits[b, s[i + n - 1]] = -float("inf")
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
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
        bs = self.cfg.block_size
        idx = idx[:, -bs:]
        full = idx                                 # running sequence, for anti-repeat
        caches = [None] * len(self.blocks)
        pos = 0
        cur = idx
        for _ in range(max_new_tokens):
            T = cur.shape[1]
            if pos + T > bs:                       # reached the context limit
                return
            x = self.drop(self.tok_emb(cur))
            cos = self.rope_cos[:, :, pos:pos + T]   # RoPE at absolute positions
            sin = self.rope_sin[:, :, pos:pos + T]
            for i, blk in enumerate(self.blocks):
                x, caches[i] = blk(x, cos, sin, caches[i])
            x = self.norm(x)
            logits = self.lm_head(x[:, [-1], :])
            pos += T
            nxt = self._sample(logits, temperature, top_k, seq=full,
                               repetition_penalty=repetition_penalty,
                               no_repeat_ngram=no_repeat_ngram)
            yield nxt
            full = torch.cat((full, nxt), dim=1)
            cur = nxt                              # next step sees only the new token

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 repetition_penalty=1.0, no_repeat_ngram=0):
        """Convenience wrapper: run generate_stream to completion and return the
        full (prompt + generated) id sequence."""
        idx = idx[:, -self.cfg.block_size:]
        for nxt in self.generate_stream(idx, max_new_tokens, temperature, top_k,
                                        repetition_penalty, no_repeat_ngram):
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    @torch.no_grad()
    def generate_naive(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                       repetition_penalty=1.0, no_repeat_ngram=0):
        """Cache-free reference path: re-encodes the whole context each step, so
        it is O(T^2) but supports unbounded sliding-window generation past
        block_size. Kept mainly to validate generate()'s KV cache against it."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            nxt = self._sample(logits, temperature, top_k, seq=idx,
                               repetition_penalty=repetition_penalty,
                               no_repeat_ngram=no_repeat_ngram)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
