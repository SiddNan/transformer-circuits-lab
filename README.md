# Transformer Circuits Lab

I built this to actually understand what happens inside a language model during training — not just to get a working model, but to open it up and see what's going on.

The project has two parts: a decoder-only transformer trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), and a set of interpretability tools (logit lens, attention head classification, activation patching, linear probing) that I used to analyze what the trained model learned.

---

## Why these architecture choices

I picked the same building blocks used in LLaMA/Mistral because I wanted to understand *why* those choices were made, not just copy them.

**RoPE instead of learned position embeddings** — encodes position by rotating query/key vectors in 2D subspaces. The dot product between two rotated vectors depends only on their relative angle, which means the model naturally handles relative position. Learned embeddings can't generalize beyond the training context length; RoPE can.

**RMSNorm instead of LayerNorm** — LayerNorm computes mean and variance; RMSNorm only computes variance (scales by RMS). Empirically similar performance, measurably faster. Pre-norm (applied before attention/FFN rather than after) makes the residual stream more interpretable: you can project it at any point and get meaningful predictions.

**SwiGLU activation** — uses a gating mechanism: `SwiGLU(x) = swish(W1·x) ⊙ (W3·x)`. The gate learns which features to propagate. About 10% better perplexity than ReLU at matched parameter count.

**Grouped-Query Attention** — instead of one KV head per Q head, multiple Q heads share a single KV head. This shrinks the KV-cache significantly during inference (important at scale) while losing very little quality compared to full multi-head attention.

**Residual scaling at init** — output projections are scaled by `1/√(2·n_layers)`. Without this, variance in the residual stream grows with depth at initialization, which destabilizes early training.

---

## Training

Trained on TinyStories (children's short stories, ~470M tokens after tokenization) using the GPT-2 tokenizer. The small run here used a 4.4M parameter model on CPU for 5,000 steps — enough to see real learning, not enough to generate fluent stories.

| Step | Val Loss | Val PPL |
|------|----------|---------|
| 0 | — | ~60,000 (random init) |
| 500 | 4.36 | 78.1 |
| 2500 | 3.64 | 38.0 |
| 3500 | 3.53 | 34.1 |
| 5000 | **3.46** | **32.0** |

Loss dropped from 11.1 → 3.46. PPL 32 means the model is ~1,900x better than random at predicting the next token. For a 4.4M parameter model trained on CPU, I'll take it.

For better text quality, the `small.yaml` config defines a 50M parameter model — that's the target for a proper GPU run.

---

## What I found

### Logit lens

The logit lens projects the residual stream at each layer through the unembedding matrix to see how predictions evolve. The most interesting finding was how quickly the model commits to known phrases.

**"Once upon a time" is basically a lookup:**
```
Token: ' upon' (in context "Once upon ...")
  embed:   ' upon' (1.00)  — just the token, no context yet
  layer 1: ' time' (0.32)  — already guessing the phrase
  layer 3: ' time' (0.89)  — confident
  layer 4: ' time' (0.97)  — nearly certain
  layer 5: ' time' (0.99)  — committed
```
The model resolves "once upon → time" almost entirely in the first few layers. Deeper layers don't change the prediction, they just increase confidence.

**It learned TinyStories character introduction syntax:**
```
Token: ' little' (in context "was a little")
  layer 3: ' named' (0.77)
  layer 4: ' named' (0.81)
```
TinyStories almost always introduces characters as "a little girl named X" or "a little boy named X". The model picked this up and uses it as a strong prior.

**It correctly fails on arithmetic:**
```
"1 + 1 =":
  embed:   H = 10.27  (total uncertainty — token barely exists in vocab)
  layer 6: H = 6.75   (still very uncertain)
```
Math tokens never appear in children's stories. The model doesn't hallucinate a confident wrong answer — it just stays uncertain throughout. That's actually the right behavior.

The general pattern across all prompts: entropy is high in early layers (the model is exploring) and drops in later layers (commitment). The residual stream progressively rules out wrong answers.

### Attention head analysis

Ran head classification on the prompt `"The cat sat on the mat. The cat sat on the"` — a repeated sequence designed to trigger induction behavior (if any heads have learned it).

**6 previous-token heads** found across the 6-layer model:

| Head | Score | Layer |
|------|-------|-------|
| L5 H3 | 0.552 | 5 |
| L4 H3 | 0.539 | 4 |
| L1 H3 | 0.516 | 1 |
| L2 H0 | 0.484 | 2 |
| L1 H1 | 0.477 | 1 |
| L1 H2 | 0.413 | 1 |

Previous-token heads implement a "copy the previous token" circuit. They're concentrated in layers 1-2 (early), with a couple in layers 4-5. Layer 0 has no previous-token heads at all — it seems to be doing something different, probably broad context aggregation.

**No induction heads found.** Induction heads implement the A B ... A → B in-context completion circuit described in [Olsson et al. (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). Their absence is expected — they typically emerge later in training and with longer sequences. Interesting experiment to run with more compute: at what step do they appear, and does the val loss show a phase transition at that point?

---

## Structure

```
├── model/
│   ├── config.py               # ModelConfig and TrainConfig dataclasses
│   ├── attention.py            # RoPE implementation, GQA, KV-cache
│   ├── layers.py               # RMSNorm, SwiGLU FFN, TransformerBlock
│   └── transformer.py          # Full model — forward pass exposes hidden states
│                               # and attention weights for interpretability
├── data/
│   └── dataset.py              # Packed tokenization (no padding, no wasted compute)
├── training/
│   ├── lr_schedule.py          # Cosine decay with linear warmup
│   └── trainer.py              # Mixed precision, grad accumulation, W&B, checkpointing
├── interpretability/
│   ├── logit_lens.py           # Project residual stream at each layer
│   ├── attention_patterns.py   # Classify heads: prev-token, induction, diffuse
│   ├── activation_patching.py  # Causal tracing — which components matter most
│   └── probing.py              # Linear probes on residual stream activations
├── scripts/
│   ├── train.py
│   ├── generate.py
│   └── interpret.py
└── configs/
    ├── small.yaml              # 50M params — intended for GPU
    ├── medium.yaml             # 125M params — intended for GPU
    └── mac_quick.yaml          # 4.4M params — runs on CPU/MPS
```

---

## Running it

```bash
pip install -r requirements.txt

# Train (Mac/CPU)
python scripts/train.py --config configs/mac_quick.yaml

# Train (GPU)
python scripts/train.py --config configs/small.yaml

# Generate
python scripts/generate.py \
  --checkpoint checkpoints/mac_quick/best.pt \
  --prompt "Once upon a time" \
  --max_tokens 200 --device cpu

# Interpretability
python scripts/interpret.py \
  --checkpoint checkpoints/mac_quick/best.pt \
  --experiments logit_lens attention --device cpu
```

---

## A note on how this was built

The architecture decisions, interpretability experiments, and analysis in this README are mine. I used [Claude Code](https://claude.ai/claude-code) to help write training infrastructure and boilerplate (the trainer loop, data loading, config system) — the kind of code that's necessary but not where the interesting ideas live. Think of it the way you'd use a well-documented library to handle the scaffolding so you can focus on the actual research.

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — Elhage et al., 2021
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — Olsson et al., 2022
- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) — Meng et al., 2022
