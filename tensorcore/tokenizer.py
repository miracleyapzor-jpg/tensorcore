"""BPE (Byte-Pair Encoding) tokenizer — roughly GPT-2/4 compatible.

This is a from-scratch implementation, not a wrapper around tiktoken.
The encoder/decoder are standalone and don't need any external libs.

Training a NEW tokenizer on your own data:
    tokenizer = BPETokenizer.train(texts, vocab_size=8192)
    tokenizer.save("my_tokenizer")

Loading a pre-trained one:
    tokenizer = BPETokenizer.load("my_tokenizer")

Design notes:
  - Regex-based pre-tokenization (splits on whitespace & punctuation)
    roughly matches GPT-2's split pattern
  - Special tokens: <|pad|>, <|bos|>, <|eos|>, <|unk|>
  - Saves as JSON for portability — no pickle, no protobuf
"""

import json
import regex as re
from collections import Counter, defaultdict
from typing import List, Optional, Union


# GPT-2 style pre-tokenization pattern
# Splits on: contractions ('s, 't, etc), letters, numbers, punctuation, whitespace
_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = {
    "<|pad|>": 0,
    "<|bos|>": 1,
    "<|eos|>": 2,
    "<|unk|>": 3,
}


def _pair_key(a, b):
    return (a, b)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer.

    Usage:
        tok = BPETokenizer.train(["hello world", "foo bar"], vocab_size=512)
        ids = tok.encode("hello world")
        text = tok.decode(ids)
    """

    def __init__(self):
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens = dict(SPECIAL_TOKENS)

    # ---- training ----
    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 8192,
              min_freq: int = 2, max_texts: int = 500_000):
        """Train BPE on a corpus.

        Args:
            texts: raw strings to train on. Should be decently large.
            vocab_size: target vocabulary size (includes base bytes + specials).
            min_freq: ignore pairs that appear fewer than this many times.
            max_texts: cap corpus size for memory.
        """
        if len(texts) > max_texts:
            import random
            texts = random.sample(texts, max_texts)

        tok = cls()

        # Step 1: pre-tokenize + encode as UTF-8 bytes
        splits = []
        for text in texts:
            words = _PAT.findall(text)
            for word in words:
                splits.append(list(word.encode("utf-8")))

        # Step 2: start with byte-level vocab (0-255)
        base_vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        # Step 3: count adjacent pairs
        # Boyer-Moore adjacency counting — O(N), not O(N²)
        pair_counts = Counter()
        for split in splits:
            for i in range(len(split) - 1):
                pair_counts[_pair_key(split[i], split[i + 1])] += 1

        merges = {}
        n_merges = vocab_size - len(base_vocab) - len(tok.special_tokens)

        # Step 4: iterative merging
        for _ in range(n_merges):
            if not pair_counts:
                break

            # pick most frequent pair (break ties by token id order)
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best_pair] < min_freq:
                break

            new_id = next_id
            next_id += 1
            merges[best_pair] = new_id

            # update splits — this is the expensive part
            new_splits = []
            new_counts = Counter()
            for split in splits:
                new_split = []
                i = 0
                while i < len(split):
                    if i + 1 < len(split) and _pair_key(split[i], split[i + 1]) == best_pair:
                        new_split.append(new_id)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                new_splits.append(new_split)
                for j in range(len(new_split) - 1):
                    new_counts[_pair_key(new_split[j], new_split[j + 1])] += 1

            splits = new_splits
            pair_counts = new_counts

        # Step 5: build vocab
        vocab = dict(base_vocab)
        for (a, b), tid in merges.items():
            vocab[tid] = vocab[a] + vocab[b]

        tok.vocab = vocab
        tok.merges = merges

        # decode lookup
        tok._id2bytes = {tid: b for tid, b in vocab.items()}
        tok._max_token_len = max(len(b) for b in vocab.values()) if vocab else 1

        return tok

    @classmethod
    def from_pretrained(cls, path: str):
        """Load a pre-trained tokenizer from JSON."""
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # vocab stored as {id: base64_string}
        import base64
        tok.vocab = {
            int(k): base64.b64decode(v) for k, v in data["vocab"].items()
        }
        tok.merges = {
            (int(a), int(b)): int(tid)
            for (a, b), tid in data["merges"]
        }
        tok.special_tokens = data.get("special_tokens", dict(SPECIAL_TOKENS))
        tok._id2bytes = dict(tok.vocab)
        tok._max_token_len = max(len(b) for b in tok.vocab.values())
        return tok

    def save(self, path: str):
        """Save to JSON (base64-encoded bytes for portability)."""
        import base64
        data = {
            "vocab": {str(k): base64.b64encode(v).decode("ascii") for k, v in self.vocab.items()},
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---- encode / decode ----
    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Encode text to token ids."""
        if not self.merges:
            # fallback: just encode as utf-8 bytes
            ids = list(text.encode("utf-8"))
        else:
            # pre-tokenize
            words = _PAT.findall(text)
            ids = []
            for word in words:
                tokens = list(word.encode("utf-8"))
                # greedily apply merges
                while len(tokens) >= 2:
                    best_merge = None
                    best_priority = float("inf")
                    for i in range(len(tokens) - 1):
                        pair = _pair_key(tokens[i], tokens[i + 1])
                        if pair in self.merges:
                            priority = self.merges[pair]  # lower id = learned earlier
                            if priority < best_priority:
                                best_priority = priority
                                best_merge = i
                    if best_merge is None:
                        break
                    # merge the best pair found
                    new_id = self.merges[_pair_key(tokens[best_merge], tokens[best_merge + 1])]
                    tokens = tokens[:best_merge] + [new_id] + tokens[best_merge + 2:]
                ids.extend(tokens)

        if add_special:
            ids = [self.special_tokens["<|bos|>"]] + ids + [self.special_tokens["<|eos|>"]]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids back to text."""
        special_set = set(self.special_tokens.values())
        chunks = []
        for tid in ids:
            if skip_special and tid in special_set:
                continue
            if tid in self._id2bytes:
                chunks.append(self._id2bytes[tid])
            else:
                chunks.append(self._id2bytes.get(self.special_tokens["<|unk|>"], b"?"))
        return b"".join(chunks).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    @property
    def bos_id(self):
        return self.special_tokens["<|bos|>"]

    @property
    def eos_id(self):
        return self.special_tokens["<|eos|>"]

    @property
    def pad_id(self):
        return self.special_tokens["<|pad|>"]

    def __repr__(self):
        return f"BPETokenizer(vocab={self.vocab_size}, merges={len(self.merges)})"


# ---- tiny test tokenizer ----
def toy_tokenizer():
    """Returns a tiny pre-trained tokenizer for quick testing.

    Trained on some C code & English snippets — good enough for
    debugging the model pipeline.
    """
    train_data = [
        "int main() { return 0; }",
        "float add(float a, float b) { return a + b; }",
        "for (int i = 0; i < n; i++) { arr[i] = i * i; }",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is the study of algorithms that improve through experience.",
        "Attention is all you need. The transformer architecture revolutionized NLP.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "import torch; import torch.nn as nn; model = nn.Linear(768, 10)",
        "Backpropagation computes gradients of the loss with respect to parameters.",
        "A language model predicts the probability of a sequence of tokens.",
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>",
        "struct Node { int data; struct Node* next; };",
        "void quicksort(int arr[], int low, int high) { /* partition & recurse */ }",
        "git commit -m 'initial import'; git push origin main",
        "Learning rate is the most important hyperparameter to tune.",
    ] * 200  # repeat so we have enough data for merges
    return BPETokenizer.train(train_data, vocab_size=300, min_freq=2)
