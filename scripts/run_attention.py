import torch
from src.attention import SelfAttention
from src.mha import MultiHeadAttention

torch.manual_seed(42)
torch.set_printoptions(sci_mode=False, precision=2)

B = 3
T = 4
d_model = 8
d_k = 3

# random input
x = torch.randn(B, T, d_model)

# test mha.py
mha = MultiHeadAttention(d_model=d_model, d_k=d_k, num_heads=5)
out, weights, scores = mha(x)

print("\n=== x ===")
print(x)

print("\n=== scores ===")
print(scores)

print("\n=== weights ===")
print(weights)

print("\n=== out ===")
print(out)

print("\n=== shapes ===")
print("out.shape     :", out.shape)        # (B, T, d_model)
print("weights.shape :", weights.shape)    # (B, num_heads, T, T)
print("scores.shape  :", scores.shape)     # (B, num_heads, T, T)

print("\n=== weights.sum(-1) (should be ~1.0) ===")
print(weights.sum(-1))  # (B, num_heads, T)