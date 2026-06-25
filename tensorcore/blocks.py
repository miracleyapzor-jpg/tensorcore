"""Transformer block + MLP.

SwiGLU: a gated linear unit with SiLU activation.
Proposed by Shazeer (2020) in the PaLM paper. Basically:
    SwiGLU(x) = SiLU(xW1) ⊙ (xW2)
where ⊙ is element-wise multiply. It's become the standard
activation for frontier models (Llama, Mistral, etc.).

RMSNorm: a simpler, faster LayerNorm that drops the mean
centering — just divide by RMS(x). Used in all Llama-family models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, no bias or mean subtraction. Just:
        x * weight / sqrt(mean(x²) + eps)

    Saves a few % in both memory and compute vs LayerNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # cast to float32 for numerical stability, then back
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (x_f32 * rms).to(x.dtype)
        return out * self.weight


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward with the standard 2/3 hidden ratio.

    Instead of the old FFN (2 matrices): gate & up projections
    share a single matmul in many impls, but I keep them separate
    for readability. If you're optimizing for speed, fuse them.
    """

    def __init__(self, config):
        super().__init__()
        d_ff = config.d_ff
        # SwiGLU has 3 weight matrices vs 2 in vanilla FFN
        # To keep param count comparable, d_ff is usually 2/3 of
        # what you'd use in vanilla FFN
        self.gate = nn.Linear(config.d_model, d_ff, bias=config.bias)
        self.up = nn.Linear(config.d_model, d_ff, bias=config.bias)
        self.down = nn.Linear(d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block — the standard since GPT-2.

    Layout: x -> RMSNorm -> Attention -> residual
                 -> RMSNorm -> SwiGLU   -> residual

    Pre-norm is more stable than post-norm for deep networks,
    which is why everyone switched after ~2019.
    """

    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config) if True else None  # imported at bottom
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x, kv_cache=None):
        # attention sub-block
        attn_out, new_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache)
        x = x + attn_out
        # ffn sub-block
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_cache


# ick: circular import avoidance. The attention module references
# nothing from blocks, so this is safe.
from .attention import CausalSelfAttention
