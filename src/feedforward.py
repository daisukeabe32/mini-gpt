# src/mha.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Create num_heads parallel SelfAttention modules
        self.heads = nn.ModuleList([
            SelfAttention(self.head_dim) for _ in range(num_heads)
        ])

        # Project concatenated features back to d_model
        self.linear = nn.Linear(num_heads * self.head_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        head_outputs = []   # output from each head: (B, T, d_model)
        all_weights = []    # attention weights from each head: (B, T, T)
        all_scores = []     # raw attention scores from each head: (B, T, T)

        for head in self.heads:
            out, weights, scores = head(x)
            head_outputs.append(out)
            all_weights.append(weights)
            all_scores.append(scores)

        # Concatenate along last dimension → (B, T, num_heads*d_model)
        concat = torch.cat(head_outputs, dim=-1)
        # Final linear projection back to (B, T, d_model)
        out = self.linear(concat)

        # Stack per-head attention maps into (B, num_heads, T, T)
        weights = torch.stack(all_weights, dim=1)
        scores = torch.stack(all_scores, dim=1)

        return out, weights, scores