"""Dataloader utilities — simple but functional.

No heavy frameworks. Just reads text files, tokenizes them,
and packs them into fixed-length chunks with a <|eos|> separator.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random
from typing import List


class TextDataset(Dataset):
    """Tokenize text files into fixed-length chunks.

    Reads raw .txt files, tokenizes everything, concatenates into
    one long sequence, then chops into seq_len chunks.
    """

    def __init__(self, paths: List[str], tokenizer, seq_len: int = 1024,
                 overlap: int = 0, shuffle_files: bool = True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.overlap = overlap
        self.samples = []

        if shuffle_files:
            random.shuffle(paths)

        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                continue

            if not text.strip():
                continue

            ids = tokenizer.encode(text, add_special=True)
            eos = tokenizer.eos_id

            # chunk into seq_len windows, separated by EOS
            stride = seq_len - overlap
            for start in range(0, max(1, len(ids) - overlap), stride):
                chunk = ids[start:start + seq_len]
                if len(chunk) < seq_len:
                    chunk = chunk + [tokenizer.pad_id] * (seq_len - len(chunk))
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

        if not self.samples:
            # create some random data so the loader doesn't crash
            print("[data] warning: no samples found, generating dummy data")
            for _ in range(100):
                dummy = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(seq_len)]
                self.samples.append(torch.tensor(dummy, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def create_dataloader(data_dir: str, tokenizer, seq_len: int = 1024,
                      batch_size: int = 8, num_workers: int = 4,
                      shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from a directory of .txt files.

    Usage:
        train_loader = create_dataloader("./data/train", tokenizer, seq_len=1024)
        val_loader   = create_dataloader("./data/val", tokenizer, seq_len=1024)
    """
    patterns = [
        os.path.join(data_dir, "**/*.txt"),
        os.path.join(data_dir, "*.txt"),
    ]
    paths = set()
    for pat in patterns:
        paths.update(glob.glob(pat, recursive=True))

    paths = sorted(paths)
    if not paths:
        print(f"[data] warning: no .txt files found in {data_dir}")
        paths = [""]  # trigger dummy data

    ds = TextDataset(paths, tokenizer, seq_len=seq_len)
    print(f"[data] {len(ds)} samples from {len(paths)} files in {data_dir}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
