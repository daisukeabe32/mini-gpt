import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, T, d_model)
        Returns:
          out: (B, T, d_model)
          weights: (B, num_heads, T, T)
          scores: (B, num_heads, T, T)
        """
        Q = self.W_Q(x)  # (B, T, d_k)
        K = self.W_K(x)  # (B, T, d_k)
        V = self.W_V(x)  # (B, T, d_model)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        return out, weights, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_model, d_k) for _ in range(num_heads)])  # num_heads SelfAttention heads
        self.linear = nn.Linear(d_model * num_heads, d_model)  # Linear layer to project concatenated heads back to d_model

    def forward(self, x):
        head_outputs = []   # each head's output: (B, T, d_model)
        all_scores = []     # each head's attention scores: (B, T, T)
        all_weights = []    # each head's attention weights: (B, T, T)
        for head in self.heads:
            out, weights, scores = head(x)
            head_outputs.append(out)
            all_scores.append(scores)
            all_weights.append(weights)

        concat = torch.cat(head_outputs, dim=-1)
        out = self.linear(concat)
        # stack to introduce head dimension: (B, num_heads, T, T)
        scores = torch.stack(all_scores, dim=1)
        weights = torch.stack(all_weights, dim=1)
        return out, weights, scores