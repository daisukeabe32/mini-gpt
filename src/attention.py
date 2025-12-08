import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head self-attention.
    Input shape:  (B, T, d_model)
    Output shape: (B, T, d_model)
    """

    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k

        # Linear layers for Q, K, V
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)
        """
        B, T, D = x.shape
        assert D == self.d_model

        # 1. project into Q, K, V
        #    shapes: (B, T, d_k), (B, T, d_k), (B, T, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # 2. compute attention scores
        #    Q @ K^T → (B, T, T)
        #    we transpose the last two dims of K: (..., T, d_k) -> (..., d_k, T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # 3. softmax over "key" dimension (last dim)
        weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # 4. weighted sum of V → (B, T, d_model)
        out = weights @ V

        return out, weights, scores