# src/feedforward.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used in Transformer blocks.

    Input:
        x: (B, T, d_model)

    Returns:
        out: (B, T, d_model)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 1) Project up to higher dimension d_ff
        # 2) Apply non-linearity (GELU)
        # 3) Project back down to d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x