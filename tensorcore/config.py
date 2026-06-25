"""Model configuration for TensorCore GPT.

All the knobs and dials live here. Tweak these to scale up/down.
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ModelConfig:
    # ---------- vocab ----------
    vocab_size: int = 50304  # round to 64* for nice throughput

    # ---------- architecture ----------
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4       # GQA: fewer kv heads saves memory
    d_model: int = 768
    d_ff: int = 3072          # SwiGLU intermediate (actual hidden dim ~ 2/3 of this)
    dropout: float = 0.0      # 0 = no dropout for pretraining, add for finetune
    bias: bool = False        # no bias in Linears = faster, cleaner

    # ---------- context ----------
    max_seq_len: int = 2048
    rope_theta: float = 10000.0

    # ---------- norms ----------
    norm_eps: float = 1e-5
    rms_norm: bool = True

    # ---------- activation ----------
    activation: Literal["swiglu", "gelu", "relu"] = "swiglu"

    # ---------- tying ----------
    tie_embeddings: bool = True  # weight tying: lm_head = token_embed

    # ---------- extras ----------
    use_flash: bool = False      # set True if you have flash-attn installed


# Pre-baked configs for quick experiments
def config_tiny() -> ModelConfig:
    """~15M params, great for debugging."""
    return ModelConfig(
        n_layers=6, n_heads=6, n_kv_heads=2,
        d_model=384, d_ff=1536, max_seq_len=512,
    )


def config_small() -> ModelConfig:
    """~85M params, comparable to GPT-2 small."""
    return ModelConfig(
        n_layers=12, n_heads=12, n_kv_heads=4,
        d_model=768, d_ff=3072, max_seq_len=1024,
    )


def config_medium() -> ModelConfig:
    """~350M params."""
    return ModelConfig(
        n_layers=24, n_heads=16, n_kv_heads=4,
        d_model=1024, d_ff=4096,
    )


def config_1b() -> ModelConfig:
    """~1.1B params — needs a real GPU (A100 40G or better)."""
    return ModelConfig(
        n_layers=32, n_heads=32, n_kv_heads=8,
        d_model=2048, d_ff=8192,
    )
