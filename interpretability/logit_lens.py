"""Logit Lens: visualizing how predictions form across layers.

Projects the residual stream at each layer through the unembedding matrix to
produce a probability distribution over tokens. This reveals how the model's
"belief" about the next token evolves as information flows through the layers.

Key insight: Because the model uses a pre-norm architecture with residual
connections, the residual stream at each layer is in approximately the same
"space" as the final output.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from model.transformer import Transformer


@torch.no_grad()
def logit_lens(model: Transformer, input_ids: torch.Tensor,
               positions: Optional[List[int]] = None, top_k: int = 5) -> dict:
    """Apply the logit lens to see predictions at each layer."""
    model.eval()
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    result = raw_model(input_ids, return_hidden_states=True)
    hidden_states = result["hidden_states"]

    n_layers = len(hidden_states)
    seq_len = input_ids.shape[1]
    if positions is None:
        positions = list(range(seq_len))

    all_probs, layer_predictions, layer_entropies = [], [], []

    for layer_idx, hidden in enumerate(hidden_states):
        normed = raw_model.final_norm(hidden)
        logits = raw_model.lm_head(normed)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs[0].cpu())

        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        layer_entropies.append(entropy[0].cpu())

        layer_preds = {}
        for pos in positions:
            top_vals, top_idx = probs[0, pos].topk(top_k)
            layer_preds[pos] = {"token_ids": top_idx.cpu().tolist(), "probs": top_vals.cpu().tolist()}
        layer_predictions.append({
            "layer": layer_idx,
            "label": "embed" if layer_idx == 0 else f"layer_{layer_idx}",
            "predictions": layer_preds,
        })

    return {
        "layer_predictions": layer_predictions,
        "layer_probs": torch.stack(all_probs),
        "layer_entropies": torch.stack(layer_entropies),
    }


def format_logit_lens_results(results: dict, tokenizer, input_ids: torch.Tensor,
                               positions: Optional[List[int]] = None) -> str:
    if positions is None:
        positions = list(range(input_ids.shape[1]))
    input_tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    lines = []

    for pos in positions:
        if pos >= len(input_tokens):
            continue
        lines.append(f"\n{'='*60}")
        lines.append(f"Position {pos}: '{input_tokens[pos]}' → predicting next token")
        lines.append(f"{'='*60}")
        for layer_info in results["layer_predictions"]:
            preds = layer_info["predictions"].get(pos)
            if preds is None:
                continue
            top_tokens = [f"'{tokenizer.decode([tid])}' ({p:.3f})"
                         for tid, p in zip(preds["token_ids"], preds["probs"])]
            entropy = results["layer_entropies"][layer_info["layer"], pos].item()
            lines.append(f"  {layer_info['label']:>12s} [H={entropy:.2f}]: {' | '.join(top_tokens[:3])}")

    return "\n".join(lines)


def plot_logit_lens(results: dict, tokenizer, input_ids: torch.Tensor,
                    target_token_id: Optional[int] = None, save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    probs = results["layer_probs"]
    if target_token_id is not None:
        token_probs = probs[:, :, target_token_id].numpy()
        title = f"P('{tokenizer.decode([target_token_id])}') across layers"
    else:
        targets = input_ids[0, 1:].cpu()
        n_layers, seq_len, _ = probs.shape
        token_probs = np.zeros((n_layers, seq_len - 1))
        for pos in range(seq_len - 1):
            token_probs[:, pos] = probs[:, pos, targets[pos]].numpy()
        title = "P(correct next token) across layers"

    input_tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    fig, ax = plt.subplots(figsize=(max(12, len(input_tokens) * 0.6), 8))
    im = ax.imshow(token_probs, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Position (token)")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    n_pos = token_probs.shape[1]
    if n_pos <= 40:
        ax.set_xticks(range(n_pos))
        ax.set_xticklabels(input_tokens[:n_pos], rotation=45, ha="right", fontsize=8)
    n_layers = token_probs.shape[0]
    labels = ["embed"] + [f"L{i}" for i in range(1, n_layers)]
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
