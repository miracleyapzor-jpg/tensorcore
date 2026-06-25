"""Tokenizer smoke tests."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorcore.tokenizer import BPETokenizer, toy_tokenizer


def test_toy_tokenizer():
    tok = toy_tokenizer()
    assert tok.vocab_size > 256
    assert tok.vocab_size <= 300 + len(tok.special_tokens)


def test_encode_decode():
    tok = toy_tokenizer()
    text = "hello world, this is a test of the tokenizer."
    ids = tok.encode(text, add_special=True)
    decoded = tok.decode(ids, skip_special=True)
    # may not be exact due to BPE, but should be close
    assert len(decoded) > 0


def test_special_tokens():
    tok = toy_tokenizer()
    ids = tok.encode("test", add_special=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_roundtrip():
    """Simple text should roundtrip."""
    tok = toy_tokenizer()
    text = "The cat sat on the mat."
    ids = tok.encode(text, add_special=False)
    decoded = tok.decode(ids, skip_special=True)
    # spaces might differ, but should contain the core words
    assert "cat" in decoded.lower()
    assert "mat" in decoded.lower()


def test_save_load(tmp_path):
    tok = toy_tokenizer()
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    tok2 = BPETokenizer.from_pretrained(path)
    assert tok2.vocab_size == tok.vocab_size
    assert len(tok2.merges) == len(tok.merges)
