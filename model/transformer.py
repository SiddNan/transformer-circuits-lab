"""Full transformer language model.

Ties everything together: token embeddings, the stack of transformer blocks,
final normalization, and the language model head (logit projection).

Key design decisions:
- Weight tying: input embedding and output LM head share weights
- No separate position embedding: RoPE is applied inside attention
- Hooks for interpretability: forward pass can cache intermediate residual
  stream states for later analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict

from model.config import ModelConfig
from model.layers import RMSNorm, TransformerBlock
from model.attention import precompute_rope_frequencies


class Transformer(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies (not learned parameters)
        freqs_cis = precompute_rope_frequencies(
            config.d_head, config.max_seq_len * 2, config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self._init_weights()

        # Scale residual projections by 1/sqrt(2*n_layers) for stability
        for layer in self.layers:
            scale = 1.0 / math.sqrt(2.0 * config.n_layers)
            with torch.no_grad():
                layer.attn.W_o.weight *= scale
                layer.ffn.W_down.weight *= scale

        print(f"Model initialized with {self.n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        std = self.config.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    @property
    def n_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        n -= self.lm_head.weight.numel()  # Tied weights
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        return_hidden_states: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        device = input_ids.device

        x = self.tok_emb(input_ids)
        x = self.emb_dropout(x)

        if kv_cache is not None and start_pos > 0:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        else:
            freqs_cis = self.freqs_cis[:seq_len]

        mask = None
        if return_attention:
            total_len = seq_len + (kv_cache[0][0].shape[1] if kv_cache and kv_cache[0] is not None else 0)
            mask = torch.full((seq_len, total_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=total_len - seq_len + 1)

        hidden_states = []
        attention_weights = []
        new_kv_cache = []

        if return_hidden_states:
            hidden_states.append(x.detach())

        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x, layer_new_kv, attn_w = layer(x, freqs_cis, mask, layer_kv, return_attention)
            new_kv_cache.append(layer_new_kv)
            if return_hidden_states:
                hidden_states.append(x.detach())
            if return_attention and attn_w is not None:
                attention_weights.append(attn_w.detach())

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100,
            )

        result = {"logits": logits, "kv_cache": new_kv_cache}
        if loss is not None:
            result["loss"] = loss
        if return_hidden_states:
            result["hidden_states"] = hidden_states
        if return_attention:
            result["attention_weights"] = attention_weights
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Autoregressive text generation with KV-cache."""
        self.eval()

        result = self(input_ids, kv_cache=None, start_pos=0)
        kv_cache = result["kv_cache"]
        logits = result["logits"][:, -1, :]

        generated = input_ids

        for i in range(max_new_tokens):
            logits = logits / temperature

            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_values[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            start_pos = generated.shape[1] - 1
            result = self(next_token, kv_cache=kv_cache, start_pos=start_pos)
            kv_cache = result["kv_cache"]
            logits = result["logits"][:, -1, :]

        return generated
