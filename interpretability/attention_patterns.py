"""Attention pattern analysis and head classification.

Identifies interpretable attention head behaviors:
1. Previous Token Heads: attend primarily to the immediately preceding token
2. Induction Heads: implement in-context pattern completion (A B ... A → B)
3. Entropy-based classification: sharp vs diffuse attention

References:
- Olsson et al. (2022) "In-context Learning and Induction Heads"
- Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from model.transformer import Transformer


@torch.no_grad()
def get_attention_patterns(model: Transformer, input_ids: torch.Tensor) -> List[torch.Tensor]:
    model.eval()
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    result = raw_model(input_ids, return_attention=True)
    return result["attention_weights"]


def score_previous_token_heads(attention_patterns: List[torch.Tensor]) -> np.ndarray:
    """Score each head on how much it attends to the previous token."""
    n_layers = len(attention_patterns)
    n_heads = attention_patterns[0].shape[1]
    scores = np.zeros((n_layers, n_heads))
    for layer_idx, attn in enumerate(attention_patterns):
        attn = attn[0].cpu().numpy()
        seq_len = attn.shape[-1]
        for head_idx in range(n_heads):
            scores[layer_idx, head_idx] = np.mean([
                attn[head_idx, t, t - 1] for t in range(1, seq_len)
            ])
    return scores


def score_induction_heads(attention_patterns: List[torch.Tensor],
                          input_ids: torch.Tensor) -> np.ndarray:
    """Score each head on induction behavior (A B ... A → attend to B)."""
    n_layers = len(attention_patterns)
    n_heads = attention_patterns[0].shape[1]
    scores = np.zeros((n_layers, n_heads))
    tokens = input_ids[0].cpu().numpy()
    seq_len = len(tokens)

    induction_pairs = []
    token_first = {}
    for pos in range(seq_len):
        tok = tokens[pos]
        if tok in token_first:
            if token_first[tok] + 1 < seq_len:
                induction_pairs.append((pos, token_first[tok] + 1))
        else:
            token_first[tok] = pos

    if not induction_pairs:
        return scores

    for layer_idx, attn in enumerate(attention_patterns):
        attn = attn[0].cpu().numpy()
        for head_idx in range(n_heads):
            scores[layer_idx, head_idx] = np.mean([
                attn[head_idx, qp, kp] for qp, kp in induction_pairs
            ])
    return scores


def compute_attention_entropy(attention_patterns: List[torch.Tensor]) -> np.ndarray:
    n_layers = len(attention_patterns)
    n_heads = attention_patterns[0].shape[1]
    entropies = np.zeros((n_layers, n_heads))
    for layer_idx, attn in enumerate(attention_patterns):
        attn = attn[0]
        entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)
        entropies[layer_idx] = entropy.mean(dim=-1).cpu().numpy()
    return entropies


def classify_heads(attention_patterns: List[torch.Tensor], input_ids: torch.Tensor,
                   prev_token_threshold: float = 0.4,
                   induction_threshold: float = 0.3) -> Dict[str, List[Tuple[int, int]]]:
    prev_scores = score_previous_token_heads(attention_patterns)
    ind_scores = score_induction_heads(attention_patterns, input_ids)
    entropies = compute_attention_entropy(attention_patterns)

    classification = {"previous_token": [], "induction": [], "low_entropy": [], "diffuse": []}
    n_layers, n_heads = prev_scores.shape
    median_entropy = np.median(entropies)

    for l in range(n_layers):
        for h in range(n_heads):
            if prev_scores[l, h] > prev_token_threshold:
                classification["previous_token"].append((l, h))
            elif ind_scores[l, h] > induction_threshold:
                classification["induction"].append((l, h))
            elif entropies[l, h] < median_entropy * 0.5:
                classification["low_entropy"].append((l, h))
            else:
                classification["diffuse"].append((l, h))
    return classification


def plot_attention_patterns(attention_patterns: List[torch.Tensor], tokenizer,
                            input_ids: torch.Tensor,
                            heads: Optional[List[Tuple[int, int]]] = None,
                            save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    if heads is None:
        ind_scores = score_induction_heads(attention_patterns, input_ids)
        flat_idx = np.argsort(ind_scores.ravel())[::-1][:4]
        heads = [(idx // ind_scores.shape[1], idx % ind_scores.shape[1]) for idx in flat_idx]

    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    n_show = min(len(tokens), 40)
    n_plots = len(heads)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (layer, head) in zip(axes, heads):
        attn = attention_patterns[layer][0, head, :n_show, :n_show].cpu().numpy()
        ax.imshow(attn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"L{layer}H{head}")
        if n_show <= 20:
            ax.set_xticks(range(n_show))
            ax.set_xticklabels(tokens[:n_show], rotation=90, fontsize=7)
            ax.set_yticks(range(n_show))
            ax.set_yticklabels(tokens[:n_show], fontsize=7)

    plt.suptitle("Attention Patterns", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def print_head_classification(classification: dict, prev_scores=None, ind_scores=None):
    print(f"\n{'='*60}\nATTENTION HEAD CLASSIFICATION\n{'='*60}")
    for head_type, heads in classification.items():
        if not heads:
            continue
        print(f"\n{head_type.upper()} HEADS ({len(heads)} found):")
        for l, h in heads:
            extra = ""
            if prev_scores is not None and head_type == "previous_token":
                extra = f" (score={prev_scores[l, h]:.3f})"
            elif ind_scores is not None and head_type == "induction":
                extra = f" (score={ind_scores[l, h]:.3f})"
            print(f"  Layer {l}, Head {h}{extra}")
