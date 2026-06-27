#!/usr/bin/env python
"""Gradio web demo for TensorCore GPT.

Usage:
    python scripts/app.py --checkpoint checkpoints/best.pt

Then open http://localhost:7860 in your browser.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from tensorcore.config import config_tiny
from tensorcore.model import GPT
from tensorcore.tokenizer import BPETokenizer


def load_model(checkpoint_path: str, tokenizer_path: str, device: str):
    tok = BPETokenizer.from_pretrained(tokenizer_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = ckpt["model_state"]

    cfg = config_tiny()
    cfg.vocab_size = model_state["tok_embed.weight"].shape[0]
    cfg.max_seq_len = 256

    model = GPT(cfg)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    info = {
        "params": f"{sum(p.numel() for p in model.parameters()):,}",
        "step": ckpt.get("step", "?"),
        "val_loss": f"{ckpt.get('val_loss', 0):.4f}",
        "vocab_size": cfg.vocab_size,
    }
    return model, tok, info


def create_app(model, tokenizer, device, model_info):
    try:
        import gradio as gr
    except ImportError:
        print("Install gradio: pip install gradio")
        raise

    def generate(prompt, max_tokens, temperature, top_k, top_p, seed):
        if seed > 0:
            torch.manual_seed(seed)

        ids = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([ids], device=device)

        output = model.generate(
            input_tensor,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            stop_token=tokenizer.eos_id,
        )

        generated = tokenizer.decode(output[0].tolist(), skip_special=True)
        return generated

    with gr.Blocks(title="TensorCore GPT — Shakespeare Demo") as demo:
        gr.Markdown(f"""
        # 🧠 TensorCore GPT
        ### 13M-parameter GPT trained on Shakespeare (from scratch, no frameworks)

        **Model info:** {model_info['params']} params · vocab={model_info['vocab_size']} ·
        trained {model_info['step']} steps · val_loss={model_info['val_loss']}

        Built with RoPE, GQA, SwiGLU, RMSNorm — no HuggingFace, no Transformers.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    value="First Citizen:",
                    lines=3,
                    placeholder="Enter a Shakespeare-style prompt...",
                )

                with gr.Row():
                    max_tokens = gr.Slider(16, 256, value=100, step=8, label="Max tokens")
                    temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")

                with gr.Row():
                    top_k = gr.Slider(1, 100, value=40, step=1, label="Top-K")
                    top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.02, label="Top-P")

                seed = gr.Number(value=42, label="Seed (-1 = random)", precision=0)
                btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                output = gr.Textbox(label="Generated Text", lines=15)

        with gr.Accordion("Example Prompts", open=False):
            examples = gr.Examples(
                examples=[
                    ["First Citizen:", 100, 0.8, 40, 0.9, 42],
                    ["To be or not", 120, 0.8, 40, 0.9, 42],
                    ["KING HENRY:", 100, 0.7, 50, 0.9, 123],
                    ["I love thee", 80, 0.9, 30, 0.95, 7],
                    ["My lord, I", 100, 0.8, 40, 0.9, 42],
                ],
                inputs=[prompt, max_tokens, temperature, top_k, top_p, seed],
            )

        btn.click(fn=generate, inputs=[prompt, max_tokens, temperature, top_k, top_p, seed], outputs=output)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model...")
    model, tok, info = load_model(args.checkpoint, args.tokenizer, device)
    print(f"Model: {info}")

    demo = create_app(model, tok, device, info)
    demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
