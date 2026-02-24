"""Text generation from a trained checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/small/best.pt --prompt "Once upon a time"
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import Transformer
from transformers import AutoTokenizer


def load_model(checkpoint_path: str, device: str = "cpu"):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in checkpoint["model_config"].items() if k not in ("d_head", "d_ff")}
    model_config = ModelConfig(**cfg)
    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    print(f"\nPrompt: {args.prompt}")
    print(f"Temperature: {args.temperature} | Top-k: {args.top_k} | Top-p: {args.top_p}")
    print("=" * 60)

    for i in range(args.n_samples):
        output_ids = model.generate(input_ids, max_new_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    top_k=args.top_k, top_p=args.top_p)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if args.n_samples > 1:
            print(f"\n--- Sample {i+1} ---")
        print(text)


if __name__ == "__main__":
    main()
