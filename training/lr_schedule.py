"""Learning rate schedule: linear warmup + cosine decay.

Standard schedule used in GPT-2, GPT-3, LLaMA, etc.
1. Linear warmup: LR increases linearly from 0 to max_lr over warmup_steps
2. Cosine decay: LR decreases following a cosine curve from max_lr to min_lr
"""

import math


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
