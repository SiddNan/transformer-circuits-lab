"""Core transformer building blocks: RMSNorm, SwiGLU FFN, and TransformerBlock.

Design choices explained:
- RMSNorm instead of LayerNorm: simpler (no mean subtraction), faster, and
  empirically equivalent. Used in LLaMA, Mistral, Gemma.
- Pre-norm architecture: normalize BEFORE attention/FFN, not after. This improves
  training stability and is standard in modern LLMs.
- SwiGLU activation: gated linear unit with SiLU (swish) gating. Empirically
  better than ReLU/GELU for language modeling (Shazeer 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from model.config import ModelConfig
from model.attention import GroupedQueryAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center activations (no mean subtraction).
    It only rescales by the RMS:
        RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # learned scale, initialized to 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()  # upcast to float32 for numerical stability, then cast back
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (x * self.weight).to(input_dtype)


class SwiGLUFFN(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Standard FFN:  FFN(x) = W2 * ReLU(W1 * x)
    SwiGLU FFN:    FFN(x) = W_down * (SiLU(W_gate * x) ⊙ (W_up * x))

    The gating mechanism gives the network more expressiveness per parameter.
    SiLU(x) = x * sigmoid(x), also known as "swish".
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # three matrices instead of two — the gate decides how much of W_up's output passes through
        self.W_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.W_up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.W_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W_gate(x))   # smooth gating signal
        up = self.W_up(x)               # the content to potentially pass through
        hidden = gate * up              # element-wise: gate controls what survives
        output = self.W_down(hidden)    # project back to d_model
        return self.dropout(output)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Architecture:
        x -> RMSNorm -> Attention -> + (residual) -> RMSNorm -> FFN -> + (residual)

    The residual stream is the central "highway" of information flow through the
    transformer. Each attention and FFN block reads from and writes to this stream.
    This view is central to mechanistic interpretability.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx  # useful for debugging and per-layer analysis
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        # normalize first, then attend, then add back to the residual stream
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, new_kv_cache, attn_weights = self.attn(
            x_norm, freqs_cis, mask, kv_cache, return_attention
        )
        x = residual + attn_out  # attention writes a delta onto the residual stream

        # same pattern for FFN
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out  # FFN also writes a delta, not a replacement

        return x, new_kv_cache, attn_weights
