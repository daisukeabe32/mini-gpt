import torch
from src.attention import SelfAttention

torch.manual_seed(42)
torch.set_printoptions(sci_mode=False, precision=2)

B = 1
T = 4
d_model = 8
d_k = 3

# random input
x = torch.randn(B, T, d_model)

att = SelfAttention(d_model=d_model, d_k=d_k)

out, weights, scores = att(x)

print("\n=== x ===")
print(x)

print("\n=== scores ===")
print(scores)

print("\n=== weights ===")
print(weights)

print("\n=== out ===")
print(out)