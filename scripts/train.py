#!/usr/bin/env python
"""Training entry point.

Examples:
    # Quick test on CPU (tiny model, a few steps)
    python scripts/train.py --config tiny --steps 100 --device cpu

    # Real training on a single GPU
    python scripts/train.py --config small --steps 10000 --data ./data/train

    # Resume from checkpoint
    python scripts/train.py --config medium --resume ./checkpoints/step_5000.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from tensorcore import ModelConfig, config_tiny, config_small, config_medium, config_1b
from tensorcore.model import GPT
from tensorcore.tokenizer import BPETokenizer, toy_tokenizer
from tensorcore.trainer import Trainer, TrainConfig
from tensorcore.data import create_dataloader

CONFIG_MAP = {
    "tiny": config_tiny,
    "small": config_small,
    "medium": config_medium,
    "1b": config_1b,
}


def main():
    parser = argparse.ArgumentParser(description="Train a TensorCore GPT model")
    parser.add_argument("--config", type=str, default="tiny",
                        choices=list(CONFIG_MAP.keys()),
                        help="model size preset")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--data", type=str, default="./data",
                        help="path to training data directory")
    parser.add_argument("--val-data", type=str, default=None,
                        help="path to validation data directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="./checkpoints")
    parser.add_argument("--compile", action="store_true",
                        help="enable torch.compile() for 20-30% speedup")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[main] using device: {device}")

    # ---- config ----
    model_cfg = CONFIG_MAP[args.config]()
    print(f"[main] model config: {args.config} "
          f"({model_cfg.n_layers}L, {model_cfg.d_model}d, "
          f"~{sum(0 for _ in range(1))} params)")

    # ---- tokenizer ----
    tokenizer_path = os.path.join(args.data, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"[main] loading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
    else:
        print("[main] building toy tokenizer (train your own for real use!)")
        tokenizer = toy_tokenizer()

    model_cfg.vocab_size = tokenizer.vocab_size
    print(f"[main] vocab size: {tokenizer.vocab_size}")

    # ---- model ----
    model = GPT(model_cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[main] model params: {n_params:,} (~{n_params/1e6:.1f}M)")

    # ---- data ----
    train_loader = create_dataloader(
        args.data, tokenizer,
        seq_len=model_cfg.max_seq_len,
        batch_size=args.micro_batch,
    )

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            args.val_data, tokenizer,
            seq_len=model_cfg.max_seq_len,
            batch_size=args.micro_batch,
            shuffle=False,
        )

    # ---- trainer ----
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch,
        max_steps=args.steps,
        lr_max=args.lr,
        out_dir=args.out_dir,
        use_compile=args.compile,
    )

    trainer = Trainer(model, train_cfg, tokenizer=tokenizer)

    if args.resume:
        trainer.load(args.resume)

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
