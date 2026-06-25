"""Inference helpers — prompt completion, chat, and a simple REPL.

KV-cache is built into the model.generate() method, so this
is mainly wrappers to make it nice to use.
"""

import torch
from typing import List, Optional


class CompletionEngine:
    """High-level interface for text completion.

    Usage:
        engine = CompletionEngine(model, tokenizer)
        print(engine.complete("Once upon a time", max_tokens=100))
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def complete(self, prompt: str, max_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.9, echo: bool = True) -> str:
        """Generate a completion for the given prompt."""
        ids = self.tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([ids], device=self.device)

        output = self.model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token=self.tokenizer.eos_id,
        )

        out_ids = output[0].tolist()
        if not echo:
            out_ids = out_ids[len(ids):]
        return self.tokenizer.decode(out_ids)

    def stream(self, prompt: str, max_tokens: int = 200, **kwargs):
        """Generator that yields tokens one at a time.

        Simpler than true streaming (which requires per-token forward
        passes), but works for demos.
        """
        text = self.complete(prompt, max_tokens=max_tokens, echo=False, **kwargs)
        # split into reasonable chunks for display
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]


def run_repl(engine: CompletionEngine):
    """Simple interactive REPL — Ctrl+C to exit."""
    print("\n" + "=" * 60)
    print("  TensorCore REPL  —  type a prompt & press Enter")
    print("  Ctrl+C to quit, type 'clear' to reset context")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "clear":
            print("[context cleared]\n")
            continue
        if prompt.lower() == "quit":
            break

        print()
        result = engine.complete(prompt, max_tokens=200)
        print(result)
        print()
