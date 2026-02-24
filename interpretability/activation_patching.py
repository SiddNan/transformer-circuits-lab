"""Activation Patching (Causal Tracing).

Identifies which components of the model are causally responsible for specific
predictions. From Meng et al. (2022) "Locating and Editing Factual Associations
in GPT" (the ROME paper).

Algorithm:
1. Run model on clean input, record all activations
2. Run model on corrupted input (noised embeddings)
3. For each layer/position, restore the clean activation into the corrupted
   run and measure how much this recovers the original prediction

If restoring a particular activation fully recovers the prediction, that
component is causally critical.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple

from model.transformer import Transformer


@torch.no_grad()
def get_all_activations(model: Transformer, input_ids: torch.Tensor):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.eval()
    activations = {}

    x = raw_model.tok_emb(input_ids)
    activations["embed"] = x.clone()
    seq_len = input_ids.shape[1]
    freqs_cis = raw_model.freqs_cis[:seq_len]

    for i, layer in enumerate(raw_model.layers):
        residual = x
        x_norm = layer.attn_norm(x)
        attn_out, _, _ = layer.attn(x_norm, freqs_cis)
        activations[f"attn_out_{i}"] = attn_out.clone()
        x = residual + attn_out
        activations[f"resid_post_attn_{i}"] = x.clone()

        residual = x
        x_norm = layer.ffn_norm(x)
        ffn_out = layer.ffn(x_norm)
        activations[f"ffn_out_{i}"] = ffn_out.clone()
        x = residual + ffn_out
        activations[f"resid_post_ffn_{i}"] = x.clone()

    x = raw_model.final_norm(x)
    logits = raw_model.lm_head(x)
    return activations, logits


def activation_patching(model: Transformer, clean_input_ids: torch.Tensor,
                        corrupted_input_ids: torch.Tensor,
                        target_position: int, target_token_id: int,
                        patch_type: str = "residual") -> np.ndarray:
    """Perform activation patching to identify causally important components."""
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.eval()

    seq_len = clean_input_ids.shape[1]
    n_layers = len(raw_model.layers)

    clean_acts, clean_logits = get_all_activations(raw_model, clean_input_ids)
    _, corrupt_logits = get_all_activations(raw_model, corrupted_input_ids)

    clean_prob = F.softmax(clean_logits[0, target_position], dim=-1)[target_token_id].item()
    corrupt_prob = F.softmax(corrupt_logits[0, target_position], dim=-1)[target_token_id].item()
    prob_diff = clean_prob - corrupt_prob

    if abs(prob_diff) < 1e-6:
        print("WARNING: Clean and corrupted predictions are nearly identical.")
        return np.zeros((n_layers, seq_len))

    key_prefix = {"residual": "resid_post_ffn", "attn": "attn_out", "ffn": "ffn_out"}[patch_type]
    recovery_scores = np.zeros((n_layers, seq_len))

    for layer_idx in range(n_layers):
        for pos in range(seq_len):
            patched_prob = _run_with_patch(
                raw_model, corrupted_input_ids, clean_acts,
                layer_idx, pos, f"{key_prefix}_{layer_idx}",
                target_position, target_token_id,
            )
            recovery = (patched_prob - corrupt_prob) / (prob_diff + 1e-10)
            recovery_scores[layer_idx, pos] = np.clip(recovery, -0.5, 1.5)

    return recovery_scores


def _run_with_patch(model, input_ids, clean_activations, patch_layer, patch_position,
                    patch_key, target_position, target_token_id):
    seq_len = input_ids.shape[1]
    freqs_cis = model.freqs_cis[:seq_len]
    x = model.tok_emb(input_ids)

    for i, layer in enumerate(model.layers):
        residual = x
        x_norm = layer.attn_norm(x)
        attn_out, _, _ = layer.attn(x_norm, freqs_cis)
        if patch_key == f"attn_out_{i}":
            attn_out[:, patch_position, :] = clean_activations[patch_key][:, patch_position, :]
        x = residual + attn_out

        residual = x
        x_norm = layer.ffn_norm(x)
        ffn_out = layer.ffn(x_norm)
        if patch_key == f"ffn_out_{i}":
            ffn_out[:, patch_position, :] = clean_activations[patch_key][:, patch_position, :]
        x = residual + ffn_out
        if patch_key == f"resid_post_ffn_{i}":
            x[:, patch_position, :] = clean_activations[patch_key][:, patch_position, :]

    x = model.final_norm(x)
    logits = model.lm_head(x)
    return F.softmax(logits[0, target_position], dim=-1)[target_token_id].item()


def create_corrupted_input(model: Transformer, input_ids: torch.Tensor,
                           corruption_positions: Optional[List[int]] = None) -> torch.Tensor:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    corrupted = input_ids.clone()
    seq_len = input_ids.shape[1]
    if corruption_positions is None:
        corruption_positions = list(range(seq_len - 1))
    vocab_size = raw_model.config.vocab_size
    for pos in corruption_positions:
        corrupted[0, pos] = torch.randint(0, vocab_size, (1,)).to(input_ids.device)
    return corrupted


def plot_causal_trace(recovery_scores: np.ndarray, tokenizer, input_ids: torch.Tensor,
                      title: str = "Causal Trace", save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    n_show = min(len(tokens), 40)
    fig, ax = plt.subplots(figsize=(max(12, n_show * 0.5), 6))
    im = ax.imshow(recovery_scores[:, :n_show], aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    if n_show <= 30:
        ax.set_xticks(range(n_show))
        ax.set_xticklabels(tokens[:n_show], rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, label="Recovery Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
