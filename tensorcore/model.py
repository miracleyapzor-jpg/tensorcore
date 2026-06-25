"""GPT model — the heart of tensorcore.

Architecture: GPT-2/3/4 style decoder-only transformer with:
  - Pre-norm (RMSNorm)
  - Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA)
  - SwiGLU activation
  - Optional weight tying

Forward returns raw logits. Wrap in GPTForCausalLM for
a full training/inference interface with loss & generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .blocks import TransformerBlock, RMSNorm


class GPT(nn.Module):
    """Bare GPT model. Takes token ids → returns logits.

    No LM head if tie_embeddings is True (head = embedding.weight).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head: maps d_model → vocab
        if config.tie_embeddings:
            # weight tying: the lm_head IS the embedding weight, transposed.
            # Saves ~30% of total params at small model sizes.
            self.lm_head_weight = self.tok_embed.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # init
        self.apply(self._init_weights)

        # HuggingFace compat: these are used by generate() etc
        self._keys_to_ignore_on_save = None

    def _init_weights(self, module):
        """Standard GPT-2 style init. Normal for Linears, normal for Embeddings."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets=None, kv_cache=None):
        """
        Args:
            tokens: [batch, seq_len] token ids
            targets: optional [batch, seq_len] for loss computation
            kv_cache: list of per-layer (k,v) tuples for incremental decoding
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: cross-entropy loss if targets given, else None
            new_kv_cache: list of per-layer (k,v) tuples
        """
        B, S = tokens.shape
        x = self.drop(self.tok_embed(tokens))

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[i]
            x, layer_new_cache = layer(x, kv_cache=layer_cache)
            new_kv_cache.append(layer_new_cache)

        x = self.norm(x)

        if self.config.tie_embeddings:
            logits = F.linear(x, self.lm_head_weight)
        else:
            logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, new_kv_cache

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                 stop_token: int = None):
        """Autoregressive generation with KV-cache.

        Args:
            tokens: [1, seq_len] prompt token ids
            max_new_tokens: how many to generate
            temperature: <1 = sharper, >1 = more random
            top_k: keep only top-k logits before sampling
            top_p: nucleus sampling threshold
            stop_token: if generated, stop early

        Returns:
            generated token ids [1, seq_len + new_tokens]
        """
        self.eval()
        generated = tokens.clone()
        kv_cache = None

        for _ in range(max_new_tokens):
            # forward pass: only need the last token
            if kv_cache is None:
                logits, _, kv_cache = self(generated)
            else:
                logits, _, kv_cache = self(generated[:, -1:], kv_cache=kv_cache)

            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # top-k
            if top_k > 0:
                vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < vals[:, -1:]] = float("-inf")

            # top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = next_logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                mask = cum_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()   # keep first token above threshold
                mask[:, 0] = False
                to_remove = torch.zeros_like(next_logits, dtype=torch.bool)
                to_remove.scatter_(1, sorted_idx, mask)
                next_logits[to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=1)

            if stop_token is not None and next_tok.item() == stop_token:
                break

        return generated

    def estimate_flops(self, batch_size: int = 1):
        """Rough FLOPs estimate per forward pass (no backward).

        Uses the standard 2*P per token approximation + some
        corrections for non-linear ops.
        """
        P = self.count_params()
        seq_len = self.config.max_seq_len
        # ~2*P forward FLOPs per token (matmuls dominate)
        flops_per_token = 2 * P
        return flops_per_token * batch_size * seq_len

    def count_params(self, grad_only: bool = False):
        """Total parameter count. Set grad_only=True for trainable params."""
        if grad_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class GPTForCausalLM(GPT):
    """Thin wrapper that adds HF-style interface.

    If you're using this with HuggingFace Trainer, this is what
    you want to instantiate.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """HF-compatible forward."""
        logits, loss, _ = super().forward(input_ids, targets=labels)
        if loss is not None:
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    @property
    def device(self):
        return next(self.parameters()).device
