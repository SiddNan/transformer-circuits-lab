"""Linear probing on residual stream activations.

Linear probes are small classifiers trained on frozen model activations to
detect what information is encoded at each layer. If a linear probe can
accurately classify some property from layer L's activations, that information
is "linearly accessible" at that layer.

Common probing tasks: POS tagging, NER, sentiment, token identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from torch.utils.data import TensorDataset, DataLoader

from model.transformer import Transformer


class LinearProbe(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@torch.no_grad()
def collect_activations(model: Transformer, input_ids: torch.Tensor) -> List[torch.Tensor]:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.eval()
    result = raw_model(input_ids, return_hidden_states=True)
    return result["hidden_states"]


def train_probe(activations: torch.Tensor, labels: torch.Tensor, n_classes: int,
                d_model: int, epochs: int = 20, lr: float = 1e-3,
                batch_size: int = 256, val_fraction: float = 0.2,
                device: str = "cuda") -> Tuple[LinearProbe, Dict[str, float]]:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    n_samples = activations.shape[0]
    n_val = int(n_samples * val_fraction)
    indices = torch.randperm(n_samples)

    train_acts = activations[indices[n_val:]].to(device)
    train_labels = labels[indices[n_val:]].to(device)
    val_acts = activations[indices[:n_val]].to(device)
    val_labels = labels[indices[:n_val]].to(device)

    probe = LinearProbe(d_model, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(train_acts, train_labels),
                             batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    for epoch in range(epochs):
        probe.train()
        total_loss, correct, total = 0, 0, 0
        for batch_acts, batch_labels in train_loader:
            logits = probe(batch_acts)
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_acts.shape[0]
            correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
            total += batch_acts.shape[0]

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_acts)
            val_loss = F.cross_entropy(val_logits, val_labels).item()
            val_acc = (val_logits.argmax(dim=-1) == val_labels).float().mean().item()
        best_val_acc = max(best_val_acc, val_acc)

    return probe, {"train_acc": correct/total, "val_acc": val_acc,
                   "train_loss": total_loss/total, "val_loss": val_loss,
                   "best_val_acc": best_val_acc}


def probe_all_layers(model: Transformer, probe_data: List[Tuple[torch.Tensor, torch.Tensor]],
                     n_classes: int, task_name: str = "probe",
                     device: str = "cuda") -> Dict[str, List[float]]:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    d_model = raw_model.config.d_model

    print(f"Collecting activations for probing task: {task_name}")
    all_activations = None
    all_labels = []

    for input_ids, labels in probe_data:
        input_ids = input_ids.to(device)
        hidden_states = collect_activations(raw_model, input_ids)
        batch_size, seq_len = input_ids.shape
        for layer_idx, hidden in enumerate(hidden_states):
            flat_hidden = hidden.reshape(-1, d_model).cpu()
            if all_activations is None:
                all_activations = [[] for _ in range(len(hidden_states))]
            all_activations[layer_idx].append(flat_hidden)
        all_labels.append(labels.reshape(-1).cpu())

    all_labels = torch.cat(all_labels)
    for i in range(len(all_activations)):
        all_activations[i] = torch.cat(all_activations[i])

    n_layers = len(all_activations)
    print(f"Probing {n_layers} layers with {all_labels.shape[0]} samples")

    layer_accuracies, layer_losses = [], []
    for layer_idx in range(n_layers):
        probe, metrics = train_probe(all_activations[layer_idx], all_labels,
                                     n_classes=n_classes, d_model=d_model, device=device)
        layer_accuracies.append(metrics["best_val_acc"])
        layer_losses.append(metrics["val_loss"])
        label = "embed" if layer_idx == 0 else f"layer_{layer_idx}"
        print(f"  {label}: val_acc={metrics['best_val_acc']:.3f}")

    return {"layer_accuracies": layer_accuracies, "layer_losses": layer_losses}


def create_pos_tag_data(tokenizer, n_samples: int = 5000, seq_len: int = 32,
                        device: str = "cuda") -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create synthetic probing data: detect repeated patterns."""
    print("NOTE: For real probing, use spaCy POS tags on real text.")
    data = []
    for _ in range(n_samples // 32):
        tokens = torch.randint(100, 5000, (1, seq_len))
        for pos in range(4, seq_len):
            if torch.rand(1).item() < 0.3:
                tokens[0, pos] = tokens[0, pos - 2]
        labels = torch.zeros(1, seq_len, dtype=torch.long)
        for pos in range(2, seq_len):
            if tokens[0, pos] == tokens[0, pos - 2]:
                labels[0, pos] = 1
        data.append((tokens, labels))
    return data


def plot_probing_results(results: Dict[str, List[float]], task_name: str = "Probing",
                         save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    accuracies = results["layer_accuracies"]
    n_layers = len(accuracies)
    labels = ["embed"] + [f"L{i}" for i in range(1, n_layers)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), accuracies, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title(f"{task_name}: Probe Accuracy by Layer")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
