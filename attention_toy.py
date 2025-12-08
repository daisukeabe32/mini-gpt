import torch
import torch.nn.functional as F

torch.set_printoptions(sci_mode=False, precision=2)

# 0. fix random seed for reproducibility
torch.manual_seed(42)

# 1. input tensor x
T = 4           # sequence length
d_model = 8     # embedding dimension
d_k = 3         # attention dimension (for Q, K)

x = torch.randn(T, d_model)
print("\n=== Input x (shape: {}) ===".format(x.shape))
print(x)

# 2. weight matrices
W_Q = torch.randn(d_model, d_k)
W_K = torch.randn(d_model, d_k)
W_V = torch.randn(d_model, d_model)

print("\n=== W_Q (shape: {}) ===".format(W_Q.shape))
print(W_Q)

print("\n=== W_K (shape: {}) ===".format(W_K.shape))
print(W_K)

print("\n=== W_V (shape: {}) ===".format(W_V.shape))
print(W_V)

# 3. compute Q, K, V
Q = x @ W_Q
K = x @ W_K
V = x @ W_V

print("\n=== Q (shape: {}) ===".format(Q.shape))
print(Q)

print("\n=== K (shape: {}) ===".format(K.shape))
print(K)

print("\n=== V (shape: {}) ===".format(V.shape))
print(V)

# 4. similarity scores
scores = Q @ K.T / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
print("\n=== scores (shape: {}) ===".format(scores.shape))
print(scores)

# 5. attention weights
weights = F.softmax(scores, dim=-1)
print("\n=== weights (shape: {}) ===".format(weights.shape))
print(weights)

# 6. output
out = weights @ V
print("\n=== out (shape: {}) ===".format(out.shape))
print(out)