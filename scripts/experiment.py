#!/usr/bin/env python
"""Controlled experiment: GQA head ratio vs. code/natural-language modeling.

Compares 4 attention configurations on 2 datasets:
  - MHA  (n_kv=6): full multi-head attention
  - GQA2 (n_kv=3): 2 query heads per KV head
  - GQA3 (n_kv=2): 3 query heads per KV head
  - MQA  (n_kv=1): multi-query attention (1 KV head total)

Each model: 6 layers, 6 heads, 384 dim, ~13M params, 600 steps, CPU.
"""

import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tensorcore.config import ModelConfig
from tensorcore.model import GPT
from tensorcore.tokenizer import BPETokenizer
from tensorcore.data import create_dataloader

RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_one(config_name, kv_heads, dataset_name, data_dir, tokenizer, steps=600):
    """Train a single model and return (loss_history, final_val_loss, param_count)."""
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=6, n_heads=6, n_kv_heads=kv_heads,
        d_model=384, d_ff=1536, max_seq_len=256, dropout=0.0,
    )

    device = torch.device("cpu")
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loader = create_dataloader(data_dir, tokenizer, seq_len=256, batch_size=4, num_workers=0)
    val_loader = create_dataloader(
        data_dir.replace("train", "val") if "val" in os.listdir(os.path.dirname(data_dir)) else data_dir,
        tokenizer, seq_len=256, batch_size=4, num_workers=0, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1, betas=(0.9, 0.95))
    warmup = 60
    train_iter = iter(loader)
    loss_history = []
    best_val = float("inf")

    model.train()
    for step in range(steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        tokens = batch["input_ids"]
        targets = batch["labels"]
        if tokens.shape[1] > targets.shape[1]:
            tokens = tokens[:, :targets.shape[1]]

        _, loss, _ = model(tokens, targets=targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step < warmup:
            lr = 5e-4 * step / warmup
        else:
            p = (step - warmup) / max(1, steps - warmup)
            lr = 5e-5 + 0.5 * (5e-4 - 5e-5) * (1.0 + math.cos(math.pi * p))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        loss_history.append(loss.item())

    # final validation
    model.eval()
    val_total, val_n = 0.0, 0
    with torch.no_grad():
        for vb in val_loader:
            if val_n >= 15:
                break
            vt, vl = vb["input_ids"], vb["labels"]
            if vt.shape[1] > vl.shape[1]:
                vt = vt[:, :vl.shape[1]]
            _, vl_loss, _ = model(vt, targets=vl)
            val_total += vl_loss.item()
            val_n += 1
    final_val = val_total / max(1, val_n)

    result = {
        "config": config_name,
        "kv_heads": kv_heads,
        "dataset": dataset_name,
        "params": n_params,
        "final_val_loss": round(final_val, 4),
        "final_train_loss": round(loss_history[-1], 4),
        "loss_history": [round(x, 4) for x in loss_history],
        "steps": steps,
    }
    return result


def main():
    print("=" * 60)
    print("  GQA Head Ratio Experiment")
    print("  RQ: How does KV-head count affect code vs. text modeling?")
    print("=" * 60)

    # Train tokenizers for each dataset
    tok_configs = {
        "shakespeare": "data/tokenizer.json",
        "code": "data/code_tokenizer.json",
    }

    results = []
    configs = [
        ("MHA",  6),
        ("GQA2", 3),
        ("GQA3", 2),
        ("MQA",  1),
    ]
    datasets = [
        ("shakespeare", "data/train"),
        ("code",        "data/code"),
    ]

    for ds_name, ds_dir in datasets:
        # Load or train tokenizer
        tok_path = tok_configs[ds_name]
        if not os.path.exists(tok_path):
            print(f"\nTraining tokenizer for {ds_name}...")
            with open(os.path.join(ds_dir, os.listdir(ds_dir)[0]), "r", encoding="utf-8") as f:
                lines = [l for l in f.read().split("\n") if l.strip()][:30000]
            tok = BPETokenizer.train(lines, vocab_size=800, min_freq=3)
            tok.save(tok_path)
        else:
            tok = BPETokenizer.from_pretrained(tok_path)
        print(f"\n[{ds_name}] tokenizer: vocab={tok.vocab_size}")

        for cfg_name, kv_h in configs:
            label = f"{cfg_name} on {ds_name}"
            print(f"\n  Training {label} (kv={kv_h}, 600 steps)...")
            t0 = time.time()
            r = train_one(cfg_name, kv_h, ds_name, ds_dir, tok, steps=600)
            dt = time.time() - t0
            print(f"    val_loss={r['final_val_loss']:.4f}  train_loss={r['final_train_loss']:.4f}  time={dt:.1f}s")
            results.append(r)

    # Save results
    out = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out}")

    # Quick summary table
    print("\n" + "=" * 70)
    print(f"{'Config':<10} {'Params':>10} {'Shakespeare Val':>16} {'Code Val':>16}")
    print("-" * 70)
    for cfg_name, kv_h in configs:
        s = next(r for r in results if r["config"] == cfg_name and r["dataset"] == "shakespeare")
        c = next(r for r in results if r["config"] == cfg_name and r["dataset"] == "code")
        print(f"{cfg_name:<10} {s['params']:>10,} {s['final_val_loss']:>16.4f} {c['final_val_loss']:>16.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
