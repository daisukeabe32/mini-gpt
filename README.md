# mini-gpt

A minimal implementation of the core components of a Transformer model, built for learning and experimentation.

## Contents

- `attention_toy.py`  
  A step-by-step implementation of scaled dot-product self-attention.

## Goals

- Understand self-attention deeply through code.
- Build up to a full mini-GPT model (tokenizer → embedding → multi-head attention → transformer block → language modeling head).
- Document each stage and track progress on GitHub.

## Environment

- Python 3.11 (Conda `ml` environment)
- PyTorch 2.9.1 (MPS enabled on Apple Silicon)

## Usage

Run the toy attention module:

```bash
python attention_toy.py