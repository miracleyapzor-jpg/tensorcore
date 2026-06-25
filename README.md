# TensorCore

A GPT-style language model implemented from scratch in PyTorch. Designed as a learning tool and a clean base for experimentation — not a 10k-line framework you need a week to understand.

**~800 lines of actual code.** Core model + tokenizer + training + inference. No abstraction tax.

### Why this exists

Most LLM codebases are either toy demos (single file, hardcoded hyperparams, can't actually train) or industrial frameworks (Megatron, Fairseq, etc — great but totally impenetrable if you're trying to learn). I wanted the middle ground: real techniques, readable code, actually runs.

Also I wanted full control over every piece for my own experiments without fighting someone else's abstraction layers.

## Architecture

Standard GPT-style decoder-only transformer, but with techniques from the post-GPT-3 era:

| Component | What it is | Where it's from |
|-----------|------------|-----------------|
| **RMSNorm** | LayerNorm minus mean-centering. Faster, same quality. | Llama / Llama 2 (2023) |
| **RoPE** | Rotary Position Embedding — position info via rotation, not addition. Extrapolates better than learned pos embeddings. | Su et al. (2021), used in Llama, Mistral, Qwen |
| **GQA** | Grouped Query Attention — fewer KV heads than Q heads. Big memory savings at inference time. | Ainslie et al. (2023), used in Llama 2 70B |
| **SwiGLU** | Gated linear unit with SiLU activation. Outperforms ReLU/GELU at same param count. | Shazeer (2020), used in PaLM, Llama |
| **Pre-norm** | Normalize before each sublayer, not after. Training stability for deep nets. | GPT-2 onward |
| **Weight tying** | LM head = embedding matrix (transposed). Saves ~30% params on small models. | Press & Wolf (2017) |

The combination of these isn't arbitrary — RoPE + GQA complement each other (RoPE makes key sharing across query heads more viable since position is encoded rotationally), and SwiGLU + RMSNorm together give you most of the Llama recipe.

### Forward pass in 3 steps

```
tokens → Embedding → [TransformerBlock × N] → RMSNorm → LM Head → logits
                           │
              ┌────────────┴────────────┐
              │  RMSNorm → Attention ──(+)──→ RMSNorm → SwiGLU ──(+)
              │     ↑ RoPE, GQA              │
              └──────────────────────────────┘
```

Each block does pre-norm attention + pre-norm FFN with residuals. That's it. The whole model is just stacking these.

## Quick Start

### Install

```bash
git clone https://github.com/YOUR_USERNAME/tensorcore.git
cd tensorcore
pip install -r requirements.txt
```

Needs PyTorch 2.0+ (for `torch.compile` support, optional) and the `regex` package for the tokenizer.

### Train a tiny model (CPU, 30 seconds)

```bash
python scripts/train.py --config tiny --steps 100 --device cpu
```

Generates dummy data automatically if you don't have a real dataset. This is just to verify everything works.

### Train on real data

Throw some `.txt` files in `./data/train/` and:

```bash
python scripts/train.py --config small --steps 10000 --data ./data/train
```

"Small" is ~85M params — big enough to learn something interesting, small enough to train on a single 3090 overnight.

### Generate text

```bash
python scripts/demo.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"
```

Or skip the prompt flag to drop into an interactive REPL.

## Project Structure

```
tensorcore/
├── tensorcore/
│   ├── config.py       # ModelConfig dataclass + presets (tiny/small/medium/1b)
│   ├── attention.py    # CausalSelfAttention + RoPE + GQA
│   ├── blocks.py       # TransformerBlock + SwiGLU + RMSNorm
│   ├── model.py        # GPT model + GPTForCausalLM wrapper
│   ├── tokenizer.py    # BPE tokenizer (train, save, load, encode, decode)
│   ├── trainer.py      # Training loop (AMP, grad accum, cosine schedule)
│   ├── data.py         # TextDataset + dataloader
│   └── inference.py    # CompletionEngine + REPL
├── scripts/
│   ├── train.py        # CLI for training
│   └── demo.py         # CLI for generation / interactive
├── tests/
│   ├── test_model.py
│   └── test_tokenizer.py
└── requirements.txt    # torch + regex + numpy
```

Everything meaningful is in `tensorcore/`. Scripts are thin CLIs.

## Design Choices I Actually Thought About

**No Flash Attention by default.** It's great, but it's a compiled C++ kernel that varies across PyTorch versions and platforms. If you have `flash-attn` installed, flip `use_flash=True` in the config. The vanilla attention is readable and correct.

**No data parallelism built in.** DDP is ~20 lines of code and tightly coupled to your launcher (`torchrun` vs `deepspeed` vs SLURM). I didn't want to pick a side. The trainer works on a single GPU and you can wrap with accelerate / deepspeed / whatever.

**Single-token targets.** The dataloader shifts input by 1 position to create targets — same as every GPT since Radford's original. No fancy span corruption or prefix-LM stuff. If you want that, `data.py` is 80 lines, fork it.

**BPE tokenizer from scratch.** Could have just wrapped tiktoken. But tokenizers are where a lot of the "magic" in LLMs actually happens, and you can't understand it if you don't build it. The implementation is a straightforward iterative merge loop with adjacency counting — same algorithm as Sennrich et al. (2016), just with bytes as base tokens.

**Cosine schedule with linear warmup.** The standard since GPT-3. Constant LR baselines do fine too but the cosine decay helps convergence in the last ~20% of training. The warmup prevents the optimizer from immediately blowing up the randomly-initialized weights.

## Scaling Reference

Rough numbers for planning experiments (A100 40G, bf16, 2048 ctx):

| Config | Params | GPU Memory | Tokens/sec | Train time (1B tok) |
|--------|--------|------------|------------|---------------------|
| tiny   | ~15M   | ~0.8 GB    | ~80k       | ~3.5 hours |
| small  | ~85M   | ~3.5 GB    | ~25k       | ~11 hours  |
| medium | ~350M  | ~14 GB     | ~8k        | ~35 hours  |
| 1b     | ~1.1B  | ~38 GB     | ~2.5k      | ~110 hours |

Memory is model + optimizer + activations. Multiply by ~4x for training (Adam states). The 1B config needs gradient accumulation + CPU offload to fit in 40GB.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

5 smoke tests for the model (forward, loss, generation, KV-cache consistency, save/load) and 5 for the tokenizer. Total runtime < 10 seconds.

## What's Missing (that I might add)

- [ ] Mixture of Experts (MoE) — the obvious next architecture knob
- [ ] Proper data preprocessing pipeline (dedup, filtering, perplexity scoring)
- [ ] Wandb / TensorBoard integration
- [ ] LoRA / QLoRA for fine-tuning
- [ ] Quantization (GPTQ / AWQ / bitsandbytes)
- [ ] Speculative decoding
- [ ] Training on OpenWebText / C4 / the Pile

Some of these are easy to add (Wandb is like 5 lines), others are real projects. PRs welcome if you've got something working.

## References

The main papers this code draws from:

- *Attention Is All You Need* (Vaswani et al., 2017)
- *Language Models are Unsupervised Multitask Learners* — GPT-2 (Radford et al., 2019)
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021)
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (Ainslie et al., 2023)
- *GLU Variants Improve Transformer* (Shazeer, 2020)
- *Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019)
- *Using the Output Embedding to Improve Language Models* — weight tying (Press & Wolf, 2017)
- *Llama 2: Open Foundation and Fine-Tuned Chat Models* (Touvron et al., 2023)

## License

MIT — do whatever, just don't blame me if it doesn't converge.
