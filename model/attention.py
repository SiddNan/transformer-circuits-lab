"""Multi-head attention with Rotary Positional Embeddings and KV-cache.

Key concepts implemented here:
- Rotary Positional Embeddings (RoPE): encodes position by rotating query/key
  vectors in 2D subspaces. This naturally captures relative position because
  the dot product of two rotated vectors depends only on their relative rotation.
- Grouped-Query Attention (GQA): uses fewer KV heads than query heads to reduce
  memory bandwidth during inference while maintaining quality.
- KV-Cache: stores past key/value tensors during autoregressive generation to
  avoid redundant computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from model.config import ModelConfig


def precompute_rope_frequencies(
    d_head: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute the complex exponentials for RoPE.

    For each dimension pair (2i, 2i+1), we compute:
        freq_i = 1 / (theta^(2i/d_head))
    Then for each position t:
        angle_{t,i} = t * freq_i

    Returns:
        freqs_cis: Complex tensor of shape (max_seq_len, d_head // 2)
                   containing e^(i * angle) for each position and dimension pair.
    """
    dim_pairs = torch.arange(0, d_head, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (dim_pairs / d_head))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings to input tensor.

    The key insight: we treat consecutive pairs of dimensions as complex numbers,
    then multiply by the precomputed complex exponentials. This rotates each 2D
    subspace by an angle proportional to the position.

    Args:
        x: Input tensor of shape (batch, seq_len, n_heads, d_head)
        freqs_cis: Precomputed frequencies of shape (seq_len, d_head // 2)

    Returns:
        Tensor with RoPE applied, same shape as input.
    """
    batch, seq_len, n_heads, d_head = x.shape
    x_complex = torch.view_as_complex(
        x.float().reshape(batch, seq_len, n_heads, d_head // 2, 2)
    )
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).reshape(batch, seq_len, n_heads, d_head)
    return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with Grouped-Query Attention (GQA) and RoPE.

    GQA uses fewer key/value heads than query heads. Each KV head is shared
    across (n_heads // n_kv_heads) query heads. This reduces memory bandwidth
    during inference (smaller KV-cache) while retaining most of MHA quality.

    When n_kv_heads == n_heads: standard Multi-Head Attention
    When n_kv_heads == 1: Multi-Query Attention
    When 1 < n_kv_heads < n_heads: Grouped-Query Attention
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.n_rep = self.n_heads // self.n_kv_heads

        self.W_q = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_k = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.W_v = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.W_o = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads."""
        if self.n_rep == 1:
            return x
        batch, seq_len, n_kv_heads, d_head = x.shape
        x = x.unsqueeze(3).expand(batch, seq_len, n_kv_heads, self.n_rep, d_head)
        return x.reshape(batch, seq_len, self.n_heads, d_head)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            freqs_cis: RoPE frequencies (seq_len, d_head // 2)
            mask: Causal mask (seq_len, total_len) or None
            kv_cache: Optional tuple of cached (keys, values) for generation
            return_attention: If True, also return attention weights

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated (keys, values) tuple
            attn_weights: Optional attention pattern (batch, n_heads, seq_len, total_len)
        """
        batch, seq_len, _ = x.shape

        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head)
        k = self.W_k(x).view(batch, seq_len, self.n_kv_heads, self.d_head)
        v = self.W_v(x).view(batch, seq_len, self.n_kv_heads, self.d_head)

        # Apply RoPE to Q and K (not V — this is important!)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # KV-cache handling for autoregressive generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        new_kv_cache = (k, v)

        # Expand KV heads for GQA
        k = self._expand_kv_heads(k)
        v = self._expand_kv_heads(v)

        # Transpose for attention: (batch, n_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        if not return_attention and mask is None:
            # Use PyTorch's optimized implementation (flash attention)
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )
            attn_weights = None
        else:
            # Manual attention computation (needed for interpretability)
            scale = 1.0 / math.sqrt(self.d_head)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn_scores = attn_scores + mask
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.W_o(output)
        output = self.resid_dropout(output)

        return output, new_kv_cache, attn_weights
