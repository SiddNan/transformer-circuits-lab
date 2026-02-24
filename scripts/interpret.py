"""Run interpretability experiments on a trained model.

Usage:
    python scripts/interpret.py --checkpoint checkpoints/small/best.pt
    python scripts/interpret.py --checkpoint checkpoints/small/best.pt --experiments logit_lens attention
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import Transformer
from transformers import AutoTokenizer

from interpretability.logit_lens import logit_lens, format_logit_lens_results, plot_logit_lens
from interpretability.attention_patterns import (
    get_attention_patterns, score_previous_token_heads, score_induction_heads,
    classify_heads, plot_attention_patterns, print_head_classification,
    compute_attention_entropy,
)
from interpretability.activation_patching import (
    activation_patching, create_corrupted_input, plot_causal_trace,
)
from interpretability.probing import probe_all_layers, create_pos_tag_data, plot_probing_results


def load_model(checkpoint_path: str, device: str = "cpu"):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in checkpoint["model_config"].items() if k not in ("d_head", "d_ff")}
    model_config = ModelConfig(**cfg)
    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_config


def run_logit_lens_experiment(model, tokenizer, device, output_dir):
    print(f"\n{'='*60}\nEXPERIMENT 1: LOGIT LENS\n{'='*60}")
    prompts = [
        "Once upon a time there was a little",
        "The cat sat on the",
        "She went to the store to buy some",
        "1 + 1 =",
    ]
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        results = logit_lens(model, input_ids, top_k=5)
        print(format_logit_lens_results(results, tokenizer, input_ids))
        plot_logit_lens(results, tokenizer, input_ids,
                       save_path=os.path.join(output_dir, f"logit_lens_{i}.png"))


def run_attention_experiment(model, tokenizer, device, output_dir):
    print(f"\n{'='*60}\nEXPERIMENT 2: ATTENTION HEAD ANALYSIS\n{'='*60}")
    prompt = "The cat sat on the mat. The cat sat on the"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt: '{prompt}'")

    attn_patterns = get_attention_patterns(model, input_ids)
    prev_scores = score_previous_token_heads(attn_patterns)
    ind_scores = score_induction_heads(attn_patterns, input_ids)

    classification = classify_heads(attn_patterns, input_ids)
    print_head_classification(classification, prev_scores, ind_scores)
    plot_attention_patterns(attn_patterns, tokenizer, input_ids,
                          save_path=os.path.join(output_dir, "attention_induction_heads.png"))

    n_heads = prev_scores.shape[1]
    flat_idx = prev_scores.ravel().argsort()[::-1][:4]
    prev_heads = [(idx // n_heads, idx % n_heads) for idx in flat_idx]
    plot_attention_patterns(attn_patterns, tokenizer, input_ids, heads=prev_heads,
                          save_path=os.path.join(output_dir, "attention_prev_token_heads.png"))


def run_causal_tracing_experiment(model, tokenizer, device, output_dir):
    print(f"\n{'='*60}\nEXPERIMENT 3: ACTIVATION PATCHING (CAUSAL TRACING)\n{'='*60}")
    prompt = "The cat sat on the mat"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    with torch.no_grad():
        result = raw_model(input_ids)
        top_token = result["logits"][0, -1].argmax().item()
    print(f"Prompt: '{prompt}' → predicts '{tokenizer.decode([top_token])}'")

    corrupted_ids = create_corrupted_input(model, input_ids)

    for patch_type in ["residual", "attn", "ffn"]:
        print(f"\nPatching: {patch_type}")
        recovery = activation_patching(model, input_ids, corrupted_ids,
                                       target_position=-1, target_token_id=top_token,
                                       patch_type=patch_type)
        plot_causal_trace(recovery, tokenizer, input_ids,
                         title=f"Causal Trace ({patch_type})",
                         save_path=os.path.join(output_dir, f"causal_trace_{patch_type}.png"))
        max_idx = recovery.argmax()
        ml, mp = max_idx // recovery.shape[1], max_idx % recovery.shape[1]
        print(f"  Most important: layer {ml}, pos {mp} (recovery={recovery[ml, mp]:.3f})")


def run_probing_experiment(model, tokenizer, device, output_dir):
    print(f"\n{'='*60}\nEXPERIMENT 4: LINEAR PROBING\n{'='*60}")
    probe_data = create_pos_tag_data(tokenizer, n_samples=2000, device=device)
    results = probe_all_layers(model, probe_data, n_classes=2,
                               task_name="pattern_detection", device=device)
    plot_probing_results(results, task_name="Pattern Detection",
                        save_path=os.path.join(output_dir, "probing_accuracy.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="interpretability_results")
    parser.add_argument("--experiments", nargs="+",
                       default=["logit_lens", "attention", "causal_tracing", "probing"],
                       choices=["logit_lens", "attention", "causal_tracing", "probing"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, config = load_model(args.checkpoint, args.device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    device = next(model.parameters()).device

    experiments = {
        "logit_lens": run_logit_lens_experiment,
        "attention": run_attention_experiment,
        "causal_tracing": run_causal_tracing_experiment,
        "probing": run_probing_experiment,
    }
    for name in args.experiments:
        experiments[name](model, tokenizer, device, args.output_dir)

    print(f"\nAll experiments complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
