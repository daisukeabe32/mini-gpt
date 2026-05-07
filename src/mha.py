# src/mha.py

import torch
import torch.nn as nn

from src.attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads

        # Create num_heads parallel SelfAttention modules
        self.heads = nn.ModuleList(
            [SelfAttention(d_model=d_model, d_k=d_k) for _ in range(num_heads)]
        )

        # Concatenate outputs from all heads: (B, T, d_k) → (B, T, num_heads*d_k)
        # Project concatenated features back to d_model
        self.W_O = nn.Linear(num_heads * d_k, d_model)
        

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)

        Returns:
          out:       (B, T, d_model)
          weights:   (B, num_heads, T, T)
          scores:    (B, num_heads, T, T)
        """
        head_outputs = []   # output from each head: (B, T, d_k)
        all_weights = []    # attention weights from each head: (B, T, T)
        all_scores = []     # raw attention scores from each head: (B, T, T)

        for head in self.heads:
            out_h, weights_h, scores_h = head(x)
            head_outputs.append(out_h)
            all_weights.append(weights_h)
            all_scores.append(scores_h)

        # Concatenate along last dimension → (B, T, num_heads*d_k)
        concat = torch.cat(head_outputs, dim=-1)

        # Final linear projection back to (B, T, d_model)
        out = self.W_O(concat)

        # Stack per-head attention maps into (B, num_heads, T, T)
        weights = torch.stack(all_weights, dim=1)
        scores = torch.stack(all_scores, dim=1)

        return out, weights, scores