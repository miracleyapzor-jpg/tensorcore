#!/usr/bin/env python
"""Quick demo — prompt the model interactively or run a one-shot completion.

Usage:
    python scripts/demo.py --checkpoint checkpoints/best.pt --prompt "Hello"

    # interactive REPL
    python scripts/demo.py --checkpoint checkpoints/best.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from tensorcore.model import GPT
from tensorcore.tokenizer import BPETokenizer, toy_tokenizer
from tensorcore.inference import CompletionEngine, run_repl
from tensorcore.config import ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Run inference with TensorCore")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="path to tokenizer JSON")
    parser.add_argument("--prompt", type=str, default=None,
                        help="single prompt to complete (skip REPL)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[demo] device: {device}")

    # ---- load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_state = ckpt["model_state"]

    # guess config from checkpoint weights
    # We save model_state, which contains the full state dict.
    # Infer architecture from the shapes:
    n_layers = sum(1 for k in model_state if k.startswith("layers.") and k.endswith(".attn_norm.weight"))
    d_model = model_state["tok_embed.weight"].shape[1]
    n_heads = 12  # default; could be inferred but it's fine
    n_kv_heads = 4

    cfg = ModelConfig(
        vocab_size=model_state["tok_embed.weight"].shape[0],
        n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
        d_model=d_model, max_seq_len=2048,
    )

    print(f"[demo] inferred config: {n_layers}L, {d_model}d, vocab={cfg.vocab_size}")

    model = GPT(cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # ---- tokenizer ----
    if args.tokenizer and os.path.exists(args.tokenizer):
        tokenizer = BPETokenizer.from_pretrained(args.tokenizer)
        print(f"[demo] tokenizer loaded: {tokenizer}")
    else:
        print("[demo] using toy tokenizer")
        tokenizer = toy_tokenizer()

    # ---- run ----
    engine = CompletionEngine(model, tokenizer, device=device)

    if args.prompt:
        result = engine.complete(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print("\n" + result)
    else:
        run_repl(engine)


if __name__ == "__main__":
    main()
