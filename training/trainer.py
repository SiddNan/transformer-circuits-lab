"""Training loop for the transformer language model.

Implements: mixed-precision training (bfloat16/float16), gradient accumulation,
gradient clipping, periodic evaluation, checkpointing, W&B logging, torch.compile.
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from typing import Optional
from pathlib import Path

from model.config import ModelConfig, TrainConfig
from model.transformer import Transformer
from data.dataset import prepare_data, create_dataloaders
from training.lr_schedule import get_lr


class Trainer:
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        self.mc = model_config
        self.tc = train_config

        device_str = train_config.device
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        elif device_str == "mps" and not torch.backends.mps.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        self.compute_dtype = dtype_map[train_config.dtype]

        if train_config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            print("WARNING: bfloat16 not supported, falling back to float16")
            self.compute_dtype = torch.float16
            train_config.dtype = "float16"

        self.use_scaler = (self.compute_dtype == torch.float16)
        self.scaler = GradScaler("cuda", enabled=self.use_scaler)
        print(f"Device: {self.device} | Compute dtype: {self.compute_dtype}")

        self.model = Transformer(model_config).to(self.device)

        if train_config.compile and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        self.optimizer = self._create_optimizer()
        os.makedirs(train_config.output_dir, exist_ok=True)

        self.wandb_run = None
        if train_config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=train_config.wandb_project,
                    name=train_config.wandb_run_name,
                    config={"model": vars(model_config), "training": vars(train_config)},
                )
            except ImportError:
                print("wandb not installed, skipping logging")

    def _create_optimizer(self) -> torch.optim.AdamW:
        decay_params, nodecay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "norm" in name or "bias" in name:
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer: {n_decay:,} params with decay, {n_nodecay:,} without")

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.tc.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.tc.learning_rate,
            betas=(self.tc.beta1, self.tc.beta2),
            eps=self.tc.eps,
            fused=torch.cuda.is_available(),
        )

    @torch.no_grad()
    def evaluate(self, val_loader) -> dict:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for i, (input_ids, targets) in enumerate(val_loader):
            if i >= self.tc.eval_steps:
                break
            input_ids, targets = input_ids.to(self.device), targets.to(self.device)
            with autocast(device_type="cuda", dtype=self.compute_dtype, enabled=self.device.type == "cuda"):
                result = self.model(input_ids, targets=targets)
            total_loss += result["loss"].item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        self.model.train()
        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def save_checkpoint(self, step: int, val_loss: float, is_best: bool = False):
        raw_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        checkpoint = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": vars(self.mc),
            "train_config": vars(self.tc),
            "step": step,
            "val_loss": val_loss,
        }
        path = os.path.join(self.tc.output_dir, f"checkpoint_step{step}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        if is_best:
            best_path = os.path.join(self.tc.output_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model! val_loss={val_loss:.4f}")

    def train(self):
        print(f"\n{'='*60}\nStarting training\n"
              f"Effective batch size: {self.tc.batch_size * self.tc.gradient_accumulation_steps}\n"
              f"Total steps: {self.tc.max_steps}\n{'='*60}\n")

        train_dataset, val_dataset, tokenizer = prepare_data(
            dataset_name=self.tc.dataset_name, seq_len=self.tc.seq_len,
        )
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, batch_size=self.tc.batch_size,
        )

        self.model.train()
        best_val_loss = float("inf")
        step, tokens_processed = 0, 0

        def infinite_loader():
            while True:
                yield from train_loader

        train_iter = iter(infinite_loader())
        t_start = time.time()

        while step < self.tc.max_steps:
            lr = get_lr(step, self.tc.learning_rate, self.tc.min_lr,
                       self.tc.warmup_steps, self.tc.max_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(self.tc.gradient_accumulation_steps):
                input_ids, targets = next(train_iter)
                input_ids, targets = input_ids.to(self.device), targets.to(self.device)
                with autocast(device_type="cuda", dtype=self.compute_dtype, enabled=self.device.type == "cuda"):
                    result = self.model(input_ids, targets=targets)
                    loss = result["loss"] / self.tc.gradient_accumulation_steps
                accum_loss += loss.item()
                self.scaler.scale(loss).backward()
                tokens_processed += input_ids.numel()

            if self.tc.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
            else:
                grad_norm = None

            self.scaler.step(self.optimizer)
            self.scaler.update()
            step += 1

            if step % self.tc.log_interval == 0:
                dt = time.time() - t_start
                tps = tokens_processed / dt
                ppl = math.exp(accum_loss) if accum_loss < 20 else float("inf")
                print(f"Step {step:>6d}/{self.tc.max_steps} | Loss: {accum_loss:.4f} | "
                      f"PPL: {ppl:.1f} | LR: {lr:.2e} | Tok/s: {tps:.0f}")
                if self.wandb_run:
                    import wandb
                    wandb.log({"train/loss": accum_loss, "train/perplexity": ppl,
                              "train/lr": lr, "train/tokens_per_sec": tps}, step=step)

            if step % self.tc.eval_interval == 0:
                metrics = self.evaluate(val_loader)
                print(f"\n  [EVAL] Step {step} | Val Loss: {metrics['val_loss']:.4f} | "
                      f"Val PPL: {metrics['val_perplexity']:.1f}\n")
                if self.wandb_run:
                    import wandb
                    wandb.log(metrics, step=step)
                is_best = metrics["val_loss"] < best_val_loss
                if is_best:
                    best_val_loss = metrics["val_loss"]
                if step % self.tc.save_interval == 0 or is_best:
                    self.save_checkpoint(step, metrics["val_loss"], is_best)

        metrics = self.evaluate(val_loader)
        self.save_checkpoint(step, metrics["val_loss"],
                           is_best=(metrics["val_loss"] < best_val_loss))
        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
        if self.wandb_run:
            import wandb
            wandb.finish()
        return best_val_loss
