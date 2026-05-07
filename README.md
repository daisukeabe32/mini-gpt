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

| Milestone | Status |
|---|---|
| M0: Causal mask | Done |
| M1: MHA refactor + TinyShakespeare + tests | Done |
| M2: Full training run, perplexity, validation set, model card | In progress |
| M3: Extension experiment (mechanistic interpretability / positional encoding study) | Planned |
| M4: arXiv preprint | Planned |
