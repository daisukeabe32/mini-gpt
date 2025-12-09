from .attention import SelfAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads

        # num_heads SelfAttention heads
        self.heads = nn.ModuleList([
            SelfAttention(d_model, d_k) for _ in range(num_heads)
        ])

        # Linear layer to project concatenated heads back to d_model
        self.W_O = nn.Linear(num_heads * d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)
        Returns:
          out: (B, T, d_model)
          weights: (B, num_heads, T, T)
          scores: (B, num_heads, T, T)
        """
        B, T, D = x.shape
        assert D == self.d_model

        head_outputs = []   # each head's output: (B, T, d_model)
        all_scores = []     # each head's attention scores: (B, T, T)
        all_weights = []    # each head's attention weights: (B, T, T)

        
        for head in self.heads:
            out_h, weights_h, scores_h = head(x)
            head_outputs.append(out_h)
            all_scores.append(scores_h)
            all_weights.append(weights_h)
        
        # (B, T, num_heads * d_model)
        concat = torch.cat(head_outputs, dim=-1)

        # stack to introduce head dimension: (B, num_heads, T, T)
        scores = torch.stack(all_scores, dim=1)
        weights = torch.stack(all_weights, dim=1)

        # (B, T, d_model)
        out = self.W_O(concat)

        return out, weights, scores