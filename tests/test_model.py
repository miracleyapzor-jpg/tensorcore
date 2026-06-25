"""Smoke tests — verify forward pass, generation, and save/load.

Run with:  python -m pytest tests/ -v
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorcore.config import config_tiny
from tensorcore.model import GPT


def test_forward():
    cfg = config_tiny()
    cfg.dropout = 0.0
    model = GPT(cfg)
    model.eval()

    tokens = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss, cache = model(tokens)

    assert logits.shape == (2, 64, cfg.vocab_size)
    assert loss is None
    assert len(cache) == cfg.n_layers


def test_loss():
    cfg = config_tiny()
    cfg.dropout = 0.0
    model = GPT(cfg)
    model.train()

    tokens = torch.randint(0, cfg.vocab_size, (2, 64))
    targets = torch.randint(0, cfg.vocab_size, (2, 64))

    logits, loss, _ = model(tokens, targets=targets)
    assert loss is not None
    assert loss.item() > 0


def test_generate():
    cfg = config_tiny()
    cfg.dropout = 0.0
    model = GPT(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)

    assert generated.shape[1] >= 10
    assert generated.shape[1] <= 10 + 20


def test_kv_cache_consistency():
    """KV-cache output should match full forward pass."""
    cfg = config_tiny()
    cfg.dropout = 0.0
    model = GPT(cfg)
    model.eval()

    tokens = torch.randint(0, cfg.vocab_size, (1, 32))

    # full forward
    logits_full, _, _ = model(tokens)
    last_full = logits_full[:, -1, :]

    # incremental: first 16 tokens, then the rest
    prefix = tokens[:, :16]
    logits_prefix, _, kv = model(prefix)

    suffix = tokens[:, 16:]
    logits_suffix, _, _ = model(suffix, kv_cache=kv)

    last_incremental = logits_suffix[:, -1, :]

    # should be close (not exact — float32 rounding)
    diff = (last_full - last_incremental).abs().max().item()
    assert diff < 1e-3, f"KV cache mismatch: max diff = {diff:.6f}"


def test_save_load(tmp_path):
    cfg = config_tiny()
    model = GPT(cfg)
    model.eval()

    # forward pass before save
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    logits_before, _, _ = model(tokens)

    # save
    path = str(tmp_path / "test.pt")
    torch.save(model.state_dict(), path)

    # load into fresh model
    model2 = GPT(cfg)
    model2.load_state_dict(torch.load(path))
    model2.eval()

    logits_after, _, _ = model2(tokens)
    assert torch.allclose(logits_before, logits_after, atol=1e-6)


def test_param_count():
    cfg = config_tiny()
    model = GPT(cfg)
    n = model.count_params()
    # tiny config should be ~32M params (mostly embedding table at 50k vocab)
    assert 20_000_000 < n < 50_000_000, f"unexpected param count: {n}"
