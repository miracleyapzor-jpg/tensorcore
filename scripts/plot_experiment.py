#!/usr/bin/env python
"""Plot experiment results comparing GQA configs across datasets."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "experiment_results/experiment_results.json"

with open(RESULTS_PATH) as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# --- Panel 1: Training curves ---
ax = axes[0]
colors = {"shakespeare": "#1a56db", "code": "#e02424"}
linestyles = {"MHA": "-", "GQA2": "--", "GQA3": "-.", "MQA": ":"}

for r in data:
    smooth = np.convolve(r["loss_history"], np.ones(20)/20, mode="valid")
    label = f"{r['config']} ({r['dataset'][:4]})"
    ax.plot(smooth, color=colors[r["dataset"]], linestyle=linestyles[r["config"]],
            alpha=0.7, linewidth=1.2, label=label)
ax.set_xlabel("Step")
ax.set_ylabel("Train Loss (smoothed)")
ax.set_title("Training Curves by GQA Configuration")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# --- Panel 2: Bar chart - final val loss ---
ax = axes[1]
configs = ["MHA", "GQA2", "GQA3", "MQA"]
x = np.arange(len(configs))
w = 0.35

shake_vals = [next(r["final_val_loss"] for r in data if r["config"]==c and r["dataset"]=="shakespeare") for c in configs]
code_vals = [next(r["final_val_loss"] for r in data if r["config"]==c and r["dataset"]=="code") for c in configs]

bars1 = ax.bar(x - w/2, shake_vals, w, label="Shakespeare (text)", color="#1a56db", edgecolor="white")
bars2 = ax.bar(x + w/2, code_vals, w, label="Python stdlib (code)", color="#e02424", edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("Validation Loss")
ax.set_title("Final Validation Loss by GQA Configuration")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 3: Parameter count vs quality ---
ax = axes[2]
params = [next(r["params"] for r in data if r["config"]==c and r["dataset"]=="shakespeare") for c in configs]

for i, c in enumerate(configs):
    sv = shake_vals[i]
    cv = code_vals[i]
    p = params[i]
    ax.scatter(p, sv, color="#1a56db", s=80, zorder=3, label="Shakespeare" if i==0 else "")
    ax.scatter(p, cv, color="#e02424", s=80, zorder=3, label="Python code" if i==0 else "")
    ax.annotate(c, (p, sv), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.annotate(c, (p, cv), textcoords="offset points", xytext=(0, -12), fontsize=8, ha="center")

ax.set_xlabel("Parameter Count")
ax.set_ylabel("Validation Loss")
ax.set_title("Params vs Quality Trade-off")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "experiment_results/experiment_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
