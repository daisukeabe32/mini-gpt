# run_block.py

import torch
from src.block import TransformerBlock


def main():
    torch.manual_seed(42)

    B, T, d_model = 2, 4, 8
    d_k = 3
    num_heads = 4
    d_ff = 32

    x = torch.randn(B, T, d_model)

    block = TransformerBlock(d_model=d_model, d_k=d_k, num_heads=num_heads, d_ff=d_ff)
    out, att_weights, att_scores = block(x)

    print("x.shape:", x.shape)
    print("out.shape:", out.shape)
    print("att_weights.shape:", att_weights.shape)
    print("att_scores.shape:", att_scores.shape)


if __name__ == "__main__":
    main()