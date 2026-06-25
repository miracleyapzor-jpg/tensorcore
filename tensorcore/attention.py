"""Multi-head attention with RoPE and Grouped Query Attention.

RoPE (Rotary Position Embedding) encodes positions by rotating
query/key vectors in pairs. GQA (Grouped Query Attention) shares
key/value heads across groups of query heads — big memory win
at negligible quality cost. Used by Llama, Mistral, DeepSeek etc.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
#  RoPE
# ---------------------------------------------------------------

def _build_rope_cache(seq_len: int, d_head: int, theta: float,
                      device: torch.device, dtype: torch.dtype):
    """Precompute cos/sin tables for RoPE.

    Rotates pairs of dims at frequencies theta^{-2i/d}.
    """
    assert d_head % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # [seq_len, d_head//2]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(x: torch.Tensor):
    """(-x2, x1, -x4, x3, ...) — the 90-degree rotation trick."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings inline — no extra memory alloc."""
    # cos, sin: [seq_len, d_head//2]
    # q, k: [B, S, n_heads, d_head]
    # Need to unsqueeze so seq_len aligns with q's S dim, and d_head//2
    # gets interleaved to d_head.
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, d_head//2]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, d_head//2]
    cos = torch.repeat_interleave(cos, 2, dim=-1)  # [1, seq, 1, d_head]
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------
#  Attention
# ---------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head / Grouped-query self-attention with causal mask.

    Uses GQA when n_kv_heads < n_heads — KV heads are repeated
    to match query heads during attention. RoPE is applied to
    Q and K *after* the projection but *before* scoring.
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_model // config.n_heads
        self.n_groups = config.n_heads // config.n_kv_heads    # q heads per kv head
        self.max_seq_len = config.max_seq_len

        # Q, K, V projections
        self.Wq = nn.Linear(config.d_model, config.n_heads * self.d_head, bias=config.bias)
        self.Wk = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=config.bias)
        self.Wv = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=config.bias)
        self.Wo = nn.Linear(config.n_heads * self.d_head, config.d_model, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # RoPE cache — lazily built
        self.rope_theta = config.rope_theta
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

        # causal mask — same deal
        self.register_buffer("causal_mask", None, persistent=False)

        self.use_flash = config.use_flash

    def _ensure_caches(self, device, dtype):
        """Build RoPE & causal mask caches on first forward or if seq_len grows."""
        if (self.rope_cos is None or self.rope_cos.device != device
                or self.rope_cos.shape[1] < self.max_seq_len):
            cos, sin = _build_rope_cache(self.max_seq_len, self.d_head,
                                         self.rope_theta, device, dtype)
            # make them contiguous so slicing is fast
            self.rope_cos = cos.contiguous()
            self.rope_sin = sin.contiguous()

        if self.causal_mask is None or self.causal_mask.shape[-1] < self.max_seq_len:
            mask = torch.triu(
                torch.full((self.max_seq_len, self.max_seq_len),
                          float("-inf"), device=device),
                diagonal=1,
            )
            self.causal_mask = mask

    def forward(self, x: torch.Tensor, kv_cache=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            kv_cache: optional tuple (k_cache, v_cache) for incremental decoding
        Returns:
            out: [batch, seq_len, d_model]
            new_cache: (k, v) tuple if kv_cache was passed, else None
        """
        B, S, _ = x.shape
        self._ensure_caches(x.device, x.dtype)
        start_pos = 0 if kv_cache is None else kv_cache[0].shape[2]

        # ---- project ----
        q = self.Wq(x).view(B, S, self.n_heads, self.d_head)
        k = self.Wk(x).view(B, S, self.n_kv_heads, self.d_head)
        v = self.Wv(x).view(B, S, self.n_kv_heads, self.d_head)

        # ---- rope ----
        cos = self.rope_cos[start_pos:start_pos + S]
        sin = self.rope_sin[start_pos:start_pos + S]
        q, k = apply_rope(q, k, cos, sin)

        # ---- transpose to [B, nh, S, d_head] ----
        q = q.transpose(1, 2)  # [B, n_heads, S, d_head]
        k = k.transpose(1, 2)  # [B, n_kv_heads, S, d_head]
        v = v.transpose(1, 2)  # [B, n_kv_heads, S, d_head]

        # ---- kv cache ----
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k, v)

        # ---- expand kv for GQA ----
        if self.n_groups > 1:
            # repeat_interleave is cleaner than repeat + reshape
            k = k.repeat_interleave(self.n_groups, dim=1)  # [B, n_heads, S_kv, d_head]
            v = v.repeat_interleave(self.n_groups, dim=1)

        # ---- score ----
        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, nh, Sq, Sk]

        # causal mask for the query positions
        if start_pos + S > 1:
            mask = self.causal_mask[start_pos:start_pos + S, :start_pos + S]
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                     # [B, nh, S, d_head]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)   # [B, S, d_model]
        return self.Wo(out), new_cache
