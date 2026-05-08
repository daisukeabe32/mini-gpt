# mini-gpt

A from-scratch implementation of a character-level GPT in PyTorch, built for deep understanding of Transformer internals.

## Architecture

```
Input (B, T)
  → CharTokenizer          character-level vocabulary
  → Token + Position Embedding
  → TransformerBlock × 4   pre-LayerNorm, causal MHA, FFN (GELU)
  → LayerNorm
  → LM Head (linear → logits)
```

**Key design choices**

- Causal self-attention with lower-triangular mask (autoregressive)
- Multi-head attention: `d_head = d_k` per head, concat → `W_O` projection
- Pre-LayerNorm (more stable than post-LN for small models)
- Training corpus: TinyShakespeare (~1M chars)

**Hyperparameters (current)**

| param | value |
|---|---|
| block\_size | 128 |
| d\_model | 64 |
| n\_layers | 4 |
| num\_heads | 4 |
| d\_k | 16 |
| d\_ff | 256 |

## Project Structure

```
mini-gpt/
├── src/
│   ├── tokenizer.py      CharTokenizer (char-level encode/decode)
│   ├── attention.py      SelfAttention with causal mask
│   ├── mha.py            MultiHeadAttention
│   ├── feedforward.py    FFN with GELU
│   ├── block.py          TransformerBlock (pre-LN + residuals)
│   └── model.py          MiniGPT (full model)
├── scripts/
│   └── train_char_gpt.py training loop + text generation
├── tests/
│   ├── test_attention.py
│   ├── test_block.py
│   ├── test_feedforward.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_training.py
└── data/
    └── shakespeare.txt
```

## Setup

```bash
conda activate ml   # Python 3.11, PyTorch 2.9.1
```

## Usage

**Train**

```bash
python -m scripts.train_char_gpt
```

Runs 5000 steps on TinyShakespeare. Prints loss every 20 steps and generates a sample at the end. Uses MPS on Apple Silicon automatically.

**Test**

```bash
python -m pytest tests/ -v
```

24 tests covering: output shapes, causal mask correctness, attention weight normalization, NaN/Inf checks, gradient flow, and an end-to-end loss-decreases integration test.

## Model Card (M2 run — visionary-vortex-4)

| | value |
|---|---|
| Parameters | 10,795,841 |
| Training corpus | TinyShakespeare (~1M chars, 90/10 split) |
| Training steps | 10,000 |
| Optimizer | AdamW, lr=3e-4 |
| Batch size | 8 |
| Block size | 256 |
| Train loss | 0.965 nats (1.39 bpc) |
| Val loss | 1.671 nats (2.41 bpc) |
| Val perplexity | 5.32 |
| Device | Apple M2 (MPS) |
| Random seed | 42 |
| W&B run | [visionary-vortex-4](https://wandb.ai/daisukeabe32-university-of-tokyo/mini-gpt/runs/fs4kn7ug) |

**Observations:** Train/val gap of 0.71 nats indicates overfitting, expected given the model size (10.8M params) relative to the corpus (~1M chars). Val loss plateaus around step 5000 and rises slightly thereafter, suggesting the optimal checkpoint is near step 5000–5500.

## Roadmap

| Milestone | Date | Status | Summary |
|---|---|---|---|
| **v0.1: Initial build** | 2025-12 | Done | Built all modules from scratch: SelfAttention → MHA → FeedForward → TransformerBlock → CharTokenizer → training loop |
| **M0: Causal mask fix** | 2026-05-04 | Done | Added lower-triangular mask to `src/attention.py`. Loss improved from 2.34 → 0.08. Autoregressive property established |
| **M1: Refactor + corpus + tests** | 2026-05-07 | Done | Canonical MHA refactor (d_head = d_k). Switched to TinyShakespeare. 24-test pytest suite |
| **M2: Full training + evaluation** | 2026-05-08 | Done | 10.8M param run, val perplexity 5.32, held-out validation, model card |
| **M3: Extension experiment** | 2026-10 | Planned | Mechanistic interpretability probe or positional encoding comparison study |
| **M4: arXiv preprint** | 2026-12 | Planned | 6-10 page writeup → arXiv (cs.LG / cs.CL) |
