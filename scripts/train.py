"""Training entrypoint.

Usage:
    python scripts/train.py --config configs/small.yaml
    python scripts/train.py --config configs/small.yaml --max_steps 10000 --no_wandb
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig, TrainConfig
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    model_config = ModelConfig.from_yaml(args.config)
    train_config = TrainConfig.from_yaml(args.config)

    if args.max_steps:  train_config.max_steps = args.max_steps
    if args.batch_size: train_config.batch_size = args.batch_size
    if args.lr:         train_config.learning_rate = args.lr
    if args.no_wandb:   train_config.use_wandb = False
    if args.no_compile: train_config.compile = False
    if args.device:     train_config.device = args.device

    print(f"{'='*60}\nMODEL: {model_config.d_model}d, {model_config.n_layers}L, "
          f"{model_config.n_heads}H | ~{model_config.n_params/1e6:.1f}M params\n"
          f"TRAINING: LR={train_config.learning_rate}, BS={train_config.batch_size}x"
          f"{train_config.gradient_accumulation_steps}, {train_config.max_steps} steps\n{'='*60}")

    trainer = Trainer(model_config, train_config)
    best_val_loss = trainer.train()
    print(f"\nDone! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
