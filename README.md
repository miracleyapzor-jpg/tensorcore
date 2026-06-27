# TensorCore

A GPT-style language model implemented from scratch in PyTorch. Designed as a learning tool and a clean base for experimentation — not a 10k-line framework you need a week to understand.

**~850 lines of actual code.** Core model + tokenizer + training + inference. No abstraction tax.

> **Live Demo:** trained on Shakespeare, running on CPU. See [Quick Start](#quick-start) to run your own.

### Why this exists

Most LLM codebases are either toy demos (single file, hardcoded hyperparams, can't actually train) or industrial frameworks (Megatron, Fairseq, etc — great but totally impenetrable if you're trying to learn). I wanted the middle ground: real techniques, readable code, actually runs.

Also I wanted full control over every piece for my own experiments without fighting someone else's abstraction layers.

## Results: 13M Model on Shakespeare

Trained a tiny (13M param) model on the Shakespeare corpus for 2000 steps on CPU (~6 minutes):

![Training Curve](checkpoints/training_curve.png)

**Loss dropped from 6.95 → 3.19 (validation), 6.95 → 2.56 (training).**

### Sample Generations

After just 2000 steps on a laptop CPU, the model produces recognizable Shakespeare-style dialogue:

| Prompt | Generation |
|--------|-----------|
| `First Citizen:` | *First Citizen: Why, that you may not / Would chosest to your gates. / BRUTUS: 'Tis awhile? / CORIOLANUS: Ay, sir...* |
| `To be or not` | *To be or notther. / COMINIUS: Nay, let's the Coriolanus. / AUFIDIUS: What is the daughter?* |
| `I love` | *I loveween together; and he, as he weddly, / Or in the winter's purpose. / DUKE VINCENTIO: You have I still free...* |

Character names (BRUTUS, CORIOLANUS, COMINIUS, MENENIUS — all actual Shakespeare characters), dialogue conventions, and Elizabethan vocabulary are all learned from scratch.

For reference: character-level models typically need 5000+ steps to get coherent. This token-level BPE model gets there faster because it learns at the subword level.

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

Each block does pre-norm attention + pre-norm FFN with residuals. That's it.

## Quick Start

### Install

```bash
git clone https://github.com/miracleyapzor-jpg/tensorcore.git
cd tensorcore
pip install -r requirements.txt
```

Needs PyTorch 2.0+ and `regex` for the tokenizer. Optional: `gradio` for the web demo, `matplotlib` for plots.

### Reproduce the Shakespeare experiment (6 min on CPU)

```bash
# 1. Download Shakespeare dataset
python -c "
import urllib.request
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
urllib.request.urlretrieve(url, 'data/train/shakespeare.txt')
"

# 2. Train BPE tokenizer
python -c "
from tensorcore.tokenizer import BPETokenizer
with open('data/train/shakespeare.txt') as f:
    lines = [l for l in f.read().split('\n') if l.strip()][:50000]
BPETokenizer.train(lines, vocab_size=1024, min_freq=3).save('data/tokenizer.json')
"

# 3. Train the model (2000 steps, ~6 min on CPU)
python scripts/train.py --config tiny --steps 2000 --device cpu --data ./data

# 4. Plot loss curve
python scripts/plot_loss.py

# 5. Generate text
python scripts/demo.py --checkpoint checkpoints/best.pt --prompt "First Citizen:"
```

### Web Demo (Gradio)

```bash
pip install gradio
python scripts/app.py --checkpoint checkpoints/best.pt
# open http://localhost:7860
```

![demo screenshot placeholder]

### Train on your own data

Throw some `.txt` files in `./data/train/` and:

```bash
# First train a tokenizer on your data
python -c "
from tensorcore.tokenizer import BPETokenizer
# ... read your text and train
"

# Then train
python scripts/train.py --config small --steps 10000 --data ./data/train
```

"Small" is ~85M params — big enough to learn something interesting, small enough to train on a single 3090 overnight.

## Project Structure

```
tensorcore/
├── tensorcore/
│   ├── config.py       # ModelConfig + presets (tiny/small/medium/1b)
│   ├── attention.py    # CausalSelfAttention + RoPE + GQA
│   ├── blocks.py       # TransformerBlock + SwiGLU + RMSNorm
│   ├── model.py        # GPT model + generate() with KV-cache
│   ├── tokenizer.py    # BPE tokenizer (train/save/load/encode/decode)
│   ├── trainer.py      # Training loop (AMP, grad accum, cosine schedule)
│   ├── data.py         # TextDataset + dataloader
│   └── inference.py    # CompletionEngine + REPL
├── scripts/
│   ├── train.py        # Training CLI
│   ├── demo.py         # Generation CLI (REPL mode)
│   ├── app.py          # Gradio web demo
│   └── plot_loss.py    # Loss curve plotter
├── tests/
│   ├── test_model.py   # 6 tests (forward, loss, generate, KV-cache, save/load, params)
│   └── test_tokenizer.py  # 5 tests (train, encode/decode, special tokens, roundtrip, save)
├── data/
│   ├── train/          # Training .txt files
│   ├── val/            # Validation .txt files
│   └── tokenizer.json  # Trained BPE tokenizer
├── checkpoints/        # Model checkpoints + training log + loss plot
└── requirements.txt
```

Everything meaningful is in `tensorcore/`. Scripts are thin CLIs.

## Design Choices

**No Flash Attention by default.** It's great, but it's a compiled C++ kernel that varies across PyTorch versions and platforms. If you have `flash-attn` installed, flip `use_flash=True` in the config. The vanilla attention is readable and correct.

**No data parallelism built in.** DDP is ~20 lines of code and tightly coupled to your launcher (`torchrun` vs `deepspeed` vs SLURM). I didn't want to pick a side. The trainer works on a single GPU and you can wrap with accelerate / deepspeed / whatever.

**Single-token targets.** The dataloader shifts input by 1 position to create targets — same as every GPT since Radford's original. No fancy span corruption or prefix-LM stuff.

**BPE tokenizer from scratch.** Could have just wrapped tiktoken. But tokenizers are where a lot of the "magic" in LLMs actually happens, and you can't understand it if you don't build it. The implementation is a straightforward iterative merge loop with adjacency counting — same algorithm as Sennrich et al. (2016), just with bytes as base tokens.

**Cosine schedule with linear warmup.** The standard since GPT-3. Constant LR baselines do fine too but the cosine decay helps convergence in the last ~20% of training. The warmup prevents the optimizer from immediately blowing up the randomly-initialized weights.

## Scaling Reference

Rough numbers for planning experiments (A100 40G, bf16, 2048 ctx):

| Config | Params | GPU Memory | Tokens/sec | Train time (1B tok) |
|--------|--------|------------|------------|---------------------|
| tiny   | ~13M   | ~0.7 GB    | ~80k       | ~3.5 hours |
| small  | ~85M   | ~3.5 GB    | ~25k       | ~11 hours  |
| medium | ~350M  | ~14 GB     | ~8k        | ~35 hours  |
| 1b     | ~1.1B  | ~38 GB     | ~2.5k      | ~110 hours |

Memory is model + optimizer + activations. Multiply by ~4x for training (Adam states). The 1B config needs gradient accumulation + CPU offload to fit in 40GB.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

11 tests: 6 for the model (forward, loss, generation, KV-cache consistency, save/load, param count) and 5 for the tokenizer. Total runtime < 10 seconds.

## What's Next

- [ ] MoE (Mixture of Experts) layers
- [ ] Wandb / TensorBoard logging
- [ ] LoRA / QLoRA for fine-tuning
- [ ] Quantization (GPTQ / AWQ)
- [ ] Speculative decoding
- [ ] Training on OpenWebText / C4 / the Pile
- [ ] RLHF / DPO alignment pipeline

PRs welcome.

## References

- *Attention Is All You Need* (Vaswani et al., 2017)
- *Language Models are Unsupervised Multitask Learners* — GPT-2 (Radford et al., 2019)
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021)
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (Ainslie et al., 2023)
- *GLU Variants Improve Transformer* (Shazeer, 2020)
- *Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019)
- *Using the Output Embedding to Improve Language Models* — weight tying (Press & Wolf, 2017)
- *Llama 2: Open Foundation and Fine-Tuned Chat Models* (Touvron et al., 2023)

## License

MIT
