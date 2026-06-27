#!/usr/bin/env python
"""Code completion benchmark: compare GQA configs on a downstream task.

Evaluates trained models on 30 Python function-completion tasks.
Metrics: syntactic validity (ast.parse), token overlap (BLEU-like),
and exact prefix match ratio.

This is NOT HumanEval — it's a lightweight proxy for a 13M model
that was trained on stdlib code, not competitive programming.
The point is the *comparison between GQA configs*, not absolute performance.
"""

import sys, os, json, ast, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import Counter

from tensorcore.config import ModelConfig
from tensorcore.model import GPT
from tensorcore.tokenizer import BPETokenizer

RESULTS_DIR = "experiment_results"

# ---- Minimal completion benchmark ----
# Each task: (prompt_with_signature, expected_body_prefix, test_input)
# The prompt is the first half of a function; the model completes the rest.
# We check if the generated code is syntactically valid Python.

BENCHMARK = [
    # --- list operations ---
    {
        "prompt": "def average(numbers):\n    \"\"\"Return the mean of a list of numbers.\"\"\"\n    ",
        "check": "return sum",
    },
    {
        "prompt": "def find_max(lst):\n    \"\"\"Find the maximum value in a list.\"\"\"\n    ",
        "check": "max_val",
    },
    {
        "prompt": "def unique_elements(items):\n    \"\"\"Return a list with duplicates removed.\"\"\"\n    ",
        "check": "list(set",
    },
    {
        "prompt": "def count_occurrences(lst, target):\n    \"\"\"Count how many times target appears in lst.\"\"\"\n    ",
        "check": "lst.count",
    },
    {
        "prompt": "def merge_lists(a, b):\n    \"\"\"Combine two lists into one sorted list.\"\"\"\n    ",
        "check": "sorted(",
    },
    # --- string operations ---
    {
        "prompt": "def is_palindrome(s):\n    \"\"\"Check if a string reads the same forward and backward.\"\"\"\n    ",
        "check": "return s",
    },
    {
        "prompt": "def capitalize_words(text):\n    \"\"\"Capitalize the first letter of each word.\"\"\"\n    ",
        "check": "text.title",
    },
    {
        "prompt": "def remove_whitespace(s):\n    \"\"\"Remove all whitespace from a string.\"\"\"\n    ",
        "check": "s.replace",
    },
    {
        "prompt": "def count_vowels(text):\n    \"\"\"Count the number of vowels in a string.\"\"\"\n    ",
        "check": "vowels",
    },
    {
        "prompt": "def reverse_string(s):\n    \"\"\"Reverse a string.\"\"\"\n    ",
        "check": "return s[::-1]",
    },
    # --- math ---
    {
        "prompt": "def factorial(n):\n    \"\"\"Compute the factorial of n.\"\"\"\n    ",
        "check": "if n",
    },
    {
        "prompt": "def is_prime(n):\n    \"\"\"Check if n is a prime number.\"\"\"\n    ",
        "check": "if n <",
    },
    {
        "prompt": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    ",
        "check": "if n",
    },
    {
        "prompt": "def gcd(a, b):\n    \"\"\"Compute the greatest common divisor using Euclidean algorithm.\"\"\"\n    ",
        "check": "while",
    },
    {
        "prompt": "def power(base, exp):\n    \"\"\"Raise base to the power exp.\"\"\"\n    ",
        "check": "return base",
    },
    # --- file / IO ---
    {
        "prompt": "def read_lines(filename):\n    \"\"\"Read all lines from a file and return as a list.\"\"\"\n    ",
        "check": "with open",
    },
    {
        "prompt": "def write_json(data, filename):\n    \"\"\"Write data as JSON to a file.\"\"\"\n    ",
        "check": "json.",
    },
    # --- dict operations ---
    {
        "prompt": "def merge_dicts(d1, d2):\n    \"\"\"Merge two dictionaries. d2 overrides d1 on conflict.\"\"\"\n    ",
        "check": "d1.copy",
    },
    {
        "prompt": "def get_keys_with_value(d, target):\n    \"\"\"Return all keys in dict d that have value == target.\"\"\"\n    ",
        "check": "[k for",
    },
    {
        "prompt": "def invert_dict(d):\n    \"\"\"Invert a dictionary: values become keys, keys become values.\"\"\"\n    ",
        "check": "new =",
    },
    # --- data processing ---
    {
        "prompt": "def filter_positive(numbers):\n    \"\"\"Return only positive numbers from the list.\"\"\"\n    ",
        "check": "[n for",
    },
    {
        "prompt": "def group_by_first_letter(words):\n    \"\"\"Group words by their first letter into a dict.\"\"\"\n    ",
        "check": "result",
    },
    {
        "prompt": "def normalize(values):\n    \"\"\"Normalize a list of numbers to [0, 1] range.\"\"\"\n    ",
        "check": "min_",
    },
    {
        "prompt": "def flatten(nested):\n    \"\"\"Flatten a list of lists into a single list.\"\"\"\n    ",
        "check": "for",
    },
    # --- class methods ---
    {
        "prompt": "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        ",
        "check": "self.items.append",
    },
    {
        "prompt": "class Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        ",
        "check": "self.count",
    },
    # --- harder: algorithms ---
    {
        "prompt": "def binary_search(arr, target):\n    \"\"\"Return index of target in sorted arr, or -1.\"\"\"\n    ",
        "check": "left",
    },
    {
        "prompt": "def bubble_sort(arr):\n    \"\"\"Sort a list using bubble sort.\"\"\"\n    ",
        "check": "for i in",
    },
    {
        "prompt": "def quicksort(arr):\n    \"\"\"Sort a list using quicksort algorithm.\"\"\"\n    ",
        "check": "if len",
    },
    {
        "prompt": "def merge_sort(arr):\n    \"\"\"Sort a list using merge sort.\"\"\"\n    ",
        "check": "mid",
    },
]


def load_model_from_checkpoint(ckpt_path, tokenizer, device="cpu"):
    """Load a trained model with its config inferred from the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state"]

    # Infer n_kv_heads from checkpoint
    kv_weight = state["layers.0.attn.Wk.weight"]
    d_model = state["tok_embed.weight"].shape[1]
    n_heads = 6
    d_head = d_model // n_heads
    n_kv_heads = kv_weight.shape[0] // d_head

    cfg_dict = {
        "vocab_size": state["tok_embed.weight"].shape[0],
        "n_layers": 6,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "d_model": d_model,
        "d_ff": 1536,
        "max_seq_len": 256,
        "dropout": 0.0,
    }

    cfg = ModelConfig(**cfg_dict)
    model = GPT(cfg)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, n_kv_heads


def is_syntactically_valid(code: str) -> bool:
    """Check if the generated code is a syntactically valid Python snippet."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def token_overlap(generated: str, reference: str) -> float:
    """Simple token overlap: Jaccard similarity of tokens."""
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    return len(gen_tokens & ref_tokens) / len(ref_tokens)


def evaluate_model(model, tokenizer, device="cpu", max_gen=80):
    """Run the benchmark and return aggregate metrics."""
    results = []
    syntax_ok = 0
    total_overlap = 0.0
    total_prefix_match = 0.0

    for task in BENCHMARK:
        ids = tokenizer.encode(task["prompt"], add_special=True)
        input_tensor = torch.tensor([ids], device=device)

        output = model.generate(
            input_tensor, max_new_tokens=max_gen, temperature=0.6,
            top_k=40, top_p=0.9, stop_token=tokenizer.eos_id,
        )

        generated = tokenizer.decode(output[0].tolist(), skip_special=True)
        # Extract the completion (everything after the prompt)
        completion = generated[len(tokenizer.decode(ids, skip_special=True)):] if len(generated) > len(task["prompt"]) else generated

        syn_ok = is_syntactically_valid(task["prompt"] + completion)
        if syn_ok:
            syntax_ok += 1

        overlap = token_overlap(completion, task["check"])
        total_overlap += overlap

        # Prefix match: does the completion start with the expected pattern?
        prefix_match = 1.0 if completion.strip().startswith(task["check"]) else 0.0
        total_prefix_match += prefix_match

        results.append({
            "prompt": task["prompt"][:50],
            "completion": completion[:100],
            "expected_prefix": task["check"],
            "prefix_match": bool(prefix_match),
            "syntax_ok": syn_ok,
            "overlap": round(overlap, 3),
        })

    n = len(BENCHMARK)
    return {
        "syntax_rate": syntax_ok / n,
        "avg_overlap": round(total_overlap / n, 4),
        "prefix_match_rate": round(total_prefix_match / n, 4),
        "per_task": results,
    }


def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """Bootstrap confidence interval for the mean."""
    data = np.array(data)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper


def main():
    device = "cpu"
    print("=" * 60)
    print("  Code Completion Benchmark: GQA Configs Compared")
    print("=" * 60)

    # Load tokenizer
    tok = BPETokenizer.from_pretrained("data/code_tokenizer.json")

    # Evaluate each model
    # Train them fresh for this evaluation (or load if we saved them)
    # Since our experiment saved results but not individual models,
    # we need to train quick evaluation models.
    # Use the experiment configs: MHA (kv=6), GQA2 (kv=3), MQA (kv=1)

    eval_configs = [
        ("MHA", 6),
        ("GQA2", 3),
        ("MQA", 1),
    ]

    all_results = {}

    for cfg_name, kv_h in eval_configs:
        print(f"\n--- {cfg_name} (n_kv={kv_h}) ---")

        # Train a quick model
        from tensorcore.config import ModelConfig
        from tensorcore.data import create_dataloader
        cfg = ModelConfig(
            vocab_size=tok.vocab_size, n_layers=6, n_heads=6,
            n_kv_heads=kv_h, d_model=384, d_ff=1536,
            max_seq_len=256, dropout=0.0,
        )
        model = GPT(cfg).to(device)

        loader = create_dataloader("data/code", tok, seq_len=256, batch_size=4, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1, betas=(0.9, 0.95))
        warmup, steps = 60, 600
        train_iter = iter(loader)

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

            if step % 200 == 0:
                print(f"  training step {step}, loss={loss.item():.4f}")

        model.eval()
        eval_result = evaluate_model(model, tok, device=device)
        all_results[cfg_name] = eval_result

        print(f"  Syntax rate:      {eval_result['syntax_rate']:.2%}")
        print(f"  Avg token overlap: {eval_result['avg_overlap']:.4f}")
        print(f"  Prefix match rate: {eval_result['prefix_match_rate']:.2%}")

    # ---- Compute statistics ----
    print("\n" + "=" * 60)
    print("  Statistical Analysis")
    print("=" * 60)

    # Paired comparison: MHA vs GQA2 on per-task overlap
    mha_overlaps = [t["overlap"] for t in all_results["MHA"]["per_task"]]
    gqa2_overlaps = [t["overlap"] for t in all_results["GQA2"]["per_task"]]
    mqa_overlaps = [t["overlap"] for t in all_results["MQA"]["per_task"]]

    # Bootstrap CIs
    for name, data in [("MHA", mha_overlaps), ("GQA2", gqa2_overlaps), ("MQA", mqa_overlaps)]:
        mean, lo, hi = bootstrap_ci(data)
        print(f"  {name}: mean overlap = {mean:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")

    # Paired differences
    diff_gqa2_mha = np.array(gqa2_overlaps) - np.array(mha_overlaps)
    mean_diff, lo, hi = bootstrap_ci(diff_gqa2_mha)
    print(f"\n  GQA2 - MHA difference: mean = {mean_diff:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")

    # Manual p-value via permutation test
    observed = np.mean(diff_gqa2_mha)
    n_perm = 5000
    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff_gqa2_mha))
        permuted_mean = np.mean(diff_gqa2_mha * signs)
        if abs(permuted_mean) >= abs(observed):
            count += 1
    p_value = count / n_perm
    print(f"  GQA2 vs MHA: permutation p-value = {p_value:.4f}")

    # ---- Syntactic validity comparison ----
    print(f"\n  Syntactic validity rates:")
    for name, res in all_results.items():
        print(f"    {name}: {res['syntax_rate']:.2%}")

    # ---- Save ----
    out_data = {
        "benchmark_description": "30 Python function completion tasks",
        "metrics": ["syntax_rate", "avg_token_overlap", "prefix_match_rate"],
        "results": {
            name: {
                "syntax_rate": r["syntax_rate"],
                "avg_overlap": r["avg_overlap"],
                "prefix_match_rate": r["prefix_match_rate"],
            }
            for name, r in all_results.items()
        },
        "statistics": {
            "bootstrap": {
                "MHA_mean_overlap": round(float(np.mean(mha_overlaps)), 4),
                "GQA2_mean_overlap": round(float(np.mean(gqa2_overlaps)), 4),
                "MQA_mean_overlap": round(float(np.mean(mqa_overlaps)), 4),
                "GQA2_MHA_diff_mean": round(float(mean_diff), 4),
                "GQA2_MHA_diff_95CI": [round(float(lo), 4), round(float(hi), 4)],
            },
            "permutation_test_p_value": round(float(p_value), 4),
        },
        "per_task_details": {name: r["per_task"] for name, r in all_results.items()},
    }

    out_path = os.path.join(RESULTS_DIR, "completion_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  KEY FINDING")
    print("=" * 60)
    if mean_diff > 0 and p_value < 0.1:
        print(f"  GQA2 outperforms MHA on code completion (d={mean_diff:.4f}, p={p_value:.3f}).")
        print(f"  This corroborates the perplexity result with a downstream task.")
    else:
        print(f"  Mixed results (d={mean_diff:.4f}, p={p_value:.3f}). Need more data.")


if __name__ == "__main__":
    main()
