#!/usr/bin/env python
"""Plot training loss curve and save as PNG."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid")


def main():
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints", "training_log.json"
    )

    if not os.path.exists(log_path):
        print(f"Log not found: {log_path}")
        print("Run training first:  python scripts/train.py --config tiny --steps 2000")
        return

    with open(log_path) as f:
        log = json.load(f)

    steps = [e["step"] for e in log]
    losses = [e["loss"] for e in log]
    lrs = [e["lr"] for e in log]
    smoothed = smooth(losses, window=20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(steps, losses, alpha=0.25, color="#1a56db", linewidth=0.5)
    ax1.plot(steps[19:], smoothed, color="#1a56db", linewidth=2, label="Smoothed (w=20)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-entropy Loss")
    ax1.set_title("Training Loss  —  TensorCore 13M on Shakespeare")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LR schedule
    ax2.plot(steps, lrs, color="#e02424", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Cosine Schedule with Linear Warmup (100 steps)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(log_path), "training_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
