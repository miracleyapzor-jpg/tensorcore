"""Training loop — mixed precision, gradient accumulation, cosine schedule.

Architecture decisions:
  - Bfloat16 autocast (less prone to overflow than float16)
  - Gradient accumulation to simulate larger batches
  - Cosine LR with linear warmup (standard since GPT-3)
  - Gradient clipping at 1.0 (stability, especially early in training)
  - Optional compile() for ~30% speedup on A100+

Usage:
    from tensorcore.trainer import Trainer
    trainer = Trainer(model, config=TrainConfig(...))
    trainer.train(train_loader, val_loader)
"""

import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class TrainConfig:
    batch_size: int = 8
    micro_batch_size: int = 2      # per-step batch (grad accum = 8/2 = 4 steps)
    max_steps: int = 10_000
    eval_every: int = 500
    save_every: int = 2_000

    # learning rate
    lr_max: float = 3e-4
    lr_min: float = 3e-5           # 10% of max — standard cosine floor
    warmup_steps: int = 500
    weight_decay: float = 0.1       # AdamW default, slightly aggressive

    # stability
    grad_clip: float = 1.0
    use_amp: bool = True            # automatic mixed precision

    # compile
    use_compile: bool = False        # torch.compile — needs PyTorch 2.0+

    # logging
    log_to_file: bool = True
    log_dir: str = "./logs"

    # checkpoint
    out_dir: str = "./checkpoints"


def _cosine_schedule(step, warmup_steps, max_steps, lr_max, lr_min):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    if step >= max_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


class Trainer:
    """Lightweight trainer — no heavy deps (no accelerate, no lightning).

    Does: AMP, grad accum, logging, checkpointing. That's it.
    """

    def __init__(self, model: nn.Module, config: TrainConfig, tokenizer=None):
        self.model = model
        self.cfg = config
        self.tokenizer = tokenizer

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr_max,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),      # GPT-3 style betas
            fused=True if torch.cuda.is_available() else False,
        )

        self.scheduler = LambdaLR(
            self.optimizer,
            lambda s: _cosine_schedule(
                s, config.warmup_steps, config.max_steps,
                config.lr_max, config.lr_min,
            ),
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
        self.step = 0
        self.best_val_loss = float("inf")

        if config.use_compile and hasattr(torch, "compile"):
            print("[trainer] compiling model with torch.compile()...")
            self.model = torch.compile(self.model)

        os.makedirs(config.out_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train(self, train_loader, val_loader=None):
        """Main training loop."""
        cfg = self.cfg
        model = self.model
        model.train()

        grad_accum = max(1, cfg.batch_size // cfg.micro_batch_size)
        print(f"[trainer] grad_accum = {grad_accum} "
              f"(batch={cfg.batch_size}, micro={cfg.micro_batch_size})")
        print(f"[trainer] {sum(p.numel() for p in model.parameters()):,} params")

        total_tokens = 0
        t0 = time.time()
        pbar = range(cfg.max_steps)

        train_iter = iter(train_loader)

        for step in pbar:
            self.step = step
            t_start = time.time()

            # ---- gradient accumulation loop ----
            accum_loss = 0.0
            for micro_step in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                tokens = batch["input_ids"]
                targets = batch.get("labels", tokens[:, 1:])
                # if we have labels, align shapes
                if tokens.shape[1] > targets.shape[1]:
                    tokens = tokens[:, :targets.shape[1]]

                with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                    _, loss, _ = model(tokens, targets=targets)

                loss = loss / grad_accum
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            # ---- optimizer step ----
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)   # set_to_none saves memory
            self.scheduler.step()

            total_tokens += tokens.numel() * grad_accum
            dt = time.time() - t_start

            # ---- logging ----
            if step % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                tok_per_sec = (tokens.numel() * grad_accum) / max(dt, 0.001)
                print(f"step {step:5d} | loss {accum_loss:.4f} | "
                      f"lr {lr:.2e} | {tok_per_sec:.0f} tok/s | "
                      f"{dt*1000:.0f}ms")

            # ---- validation ----
            if val_loader is not None and step > 0 and step % cfg.eval_every == 0:
                val_loss = self.evaluate(val_loader)
                print(f"  [eval] step {step} | val_loss {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save("best")

            # ---- checkpoint ----
            if step > 0 and step % cfg.save_every == 0:
                self.save(f"step_{step}")

        # final save
        self.save("final")
        elapsed = time.time() - t0
        print(f"[trainer] done. {self.step} steps in {elapsed/60:.1f} min. "
              f"{total_tokens/1e6:.1f}M tokens seen.")

    @torch.no_grad()
    def evaluate(self, loader, max_batches=20):
        """Compute average loss on validation set."""
        self.model.eval()
        total_loss, n = 0.0, 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            tokens = batch["input_ids"]
            targets = batch.get("labels", tokens[:, 1:])
            if tokens.shape[1] > targets.shape[1]:
                tokens = tokens[:, :targets.shape[1]]
            with torch.amp.autocast("cuda", enabled=self.cfg.use_amp):
                _, loss, _ = self.model(tokens, targets=targets)
            total_loss += loss.item()
            n += 1
        self.model.train()
        return total_loss / max(1, n)

    def save(self, tag: str):
        path = os.path.join(self.cfg.out_dir, f"{tag}.pt")
        torch.save({
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)
        print(f"  [save] {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[trainer] loaded checkpoint from step {self.step}")
