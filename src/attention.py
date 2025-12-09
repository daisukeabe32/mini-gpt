import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head self-attention.

    Input:
        x: (B, T, d_model)

    Returns:
        out:     (B, T, d_model)
        weights: (B, T, T)        attention weights (after softmax)
        scores:  (B, T, T)        raw attention scores (before softmax)
    """

    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)
        """
        B, T, D = x.shape
        assert D == self.d_model

        # Project to Q, K, V
        Q = self.W_Q(x)  # (B, T, d_k)
        K = self.W_K(x)  # (B, T, d_k)
        V = self.W_V(x)  # (B, T, d_model)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, T, T)
        weights = F.softmax(scores, dim=-1)                      # (B, T, T)
        out = weights @ V                                        # (B, T, d_model)

        return out, weights, scores