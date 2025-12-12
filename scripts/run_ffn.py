# run_ffn.py

import torch
from src.feedforward import FeedForward

torch.manual_seed(42)
torch.set_printoptions(sci_mode=False, precision=2)

B = 2
T = 4
d_model = 8
d_ff = 32  # usually 2x~4x of d_model

# random input
x = torch.randn(B, T, d_model)

ffn = FeedForward(d_model=d_model, d_ff=d_ff)
out = ffn(x)

print("x shape:   ", x.shape)
print("out shape: ", out.shape)

print("\n=== x[0] ===")
print(x[0])

print("\n=== out[0] ===")
print(out[0])