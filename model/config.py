"""Model configuration."""

from dataclasses import dataclass, field
from typing import Optional
import yaml
import math


@dataclass
class ModelConfig:
    """Configuration for the transformer language model.

    Default values correspond to a ~50M parameter model suitable
    for training on a single consumer GPU.
    """
    vocab_size: int = 32000
    d_model: int = 512        # width of the residual stream — every layer reads/writes this
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: int = 4       # GQA: fewer KV heads than Q heads, saves memory during generation
    d_ff_multiplier: float = 2.667  # SwiGLU needs ~2/3 the FFN width vs ReLU to match params
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    init_std: Optional[float] = None

    # these two are computed from the above — not passed in, just derived
    d_head: int = field(init=False)
    d_ff: int = field(init=False)

    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads
        raw_d_ff = int(self.d_model * self.d_ff_multiplier)
        # round up to nearest 64 — keeps matrix dimensions friendly for GPU kernels
        self.d_ff = ((raw_d_ff + 63) // 64) * 64
        if self.init_std is None:
            # 1/sqrt(d_model) is the standard — keeps activations from blowing up at init
            self.init_std = 1.0 / math.sqrt(self.d_model)
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def n_params(self) -> int:
        """Estimate total parameter count."""
        embed = self.vocab_size * self.d_model
        attn_q = self.d_model * self.d_model
        attn_k = self.d_model * (self.n_kv_heads * self.d_head)
        attn_v = self.d_model * (self.n_kv_heads * self.d_head)
        attn_o = self.d_model * self.d_model
        attn = attn_q + attn_k + attn_v + attn_o
        ffn = 3 * self.d_model * self.d_ff  # SwiGLU has 3 matrices (gate, up, down)
        norm = 2 * self.d_model
        block = attn + ffn + norm
        final = self.d_model
        return embed + self.n_layers * block + final

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config.get("model", config))

    def to_yaml(self, path: str):
        import dataclasses
        # exclude d_head and d_ff — they're derived, not inputs
        d = {k: v for k, v in dataclasses.asdict(self).items()
             if k not in ("d_head", "d_ff")}
        with open(path, "w") as f:
            yaml.dump({"model": d}, f, default_flow_style=False)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    learning_rate: float = 3e-4
    min_lr: float = 3e-5          # cosine decays down to this, not zero
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95           # 0.95 instead of the default 0.999 — tracks faster, better for LM
    eps: float = 1e-8
    grad_clip: float = 1.0        # clips gradient norm, not individual gradients
    batch_size: int = 32
    gradient_accumulation_steps: int = 4  # effective batch = batch_size * this
    max_steps: int = 50000
    warmup_steps: int = 1000
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 2500
    eval_steps: int = 100         # how many val batches to average over — not the full val set
    dataset_name: str = "roneneldan/TinyStories"
    seq_len: int = 1024
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True          # torch.compile can give 20-40% speedup on GPU
    output_dir: str = "checkpoints"
    wandb_project: str = "transformer-circuits-lab"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config.get("training", {}))
