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

## Roadmap

| Milestone | Date | Status | Summary |
|---|---|---|---|
| **v0.1: Initial build** | 2025-12 | Done | SelfAttention → MHA → FeedForward → TransformerBlock → CharTokenizer → training loop。全モジュール実装・結合完了 |
| **M0: Causal mask fix** | 2026-05-04 | Done | `src/attention.py` に lower-triangular mask 実装。loss 2.34 → 0.08 に改善。autoregressive property 確立 |
| **M1: Refactor + corpus + tests** | 2026-05-07 | Done | MHA canonical refactor (d_head = d_k)。TinyShakespeare 切替。pytest 24テスト体制構築 |
| **M2: Full training + evaluation** | 2026-08 | Planned | ~10M param training run、perplexity、held-out validation、model card |
| **M3: Extension experiment** | 2026-10 | Planned | mechanistic interpretability probe または positional encoding 比較研究 |
| **M4: arXiv preprint** | 2026-12 | Planned | 6-10 page writeup → arXiv (cs.LG / cs.CL) |
