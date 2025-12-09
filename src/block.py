import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import SelfAttention
from src.feedforward import FeedForward


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads

        # Create num_heads parallel SelfAttention modules
        self.heads = nn.ModuleList([SelfAttention(d_model, d_k) for _ in range(num_heads)])

        # Linear layer to combine the outputs of all heads
        self.linear = nn.Linear(num_heads * d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)

        Returns:
          out:           (B, T, d_model)
          att_weights:   (B, num_heads, T, T)
          att_scores:    (B, num_heads, T, T)
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
        att_weights = torch.stack(all_weights, dim=1)
        att_scores = torch.stack(all_scores, dim=1)

        return out, att_weights, att_scores


class TransformerBlock(nn.Module):
    """
    A single Transformer block:
      - LayerNorm + Multi-head self-attention + residual
      - LayerNorm + FeedForward + residual
    """

    def __init__(self, d_model: int, d_k: int, num_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads
        self.d_ff = d_ff

        # LayerNorm before attention and before FFN
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Multi-head self-attention (uses the class defined above)
        self.mha = MultiHeadAttention(d_model=d_model, d_k=d_k, num_heads=num_heads)

        # Position-wise feed-forward network
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, d_model)

        Returns:
            x:           (B, T, d_model)
            att_weights: (B, num_heads, T, T)
            att_scores:  (B, num_heads, T, T)
        """
        B, T, D = x.shape
        assert D == self.d_model

        # 1) LayerNorm → Multi-head attention → residual
        x_norm = self.ln1(x)
        att_out, att_weights, att_scores = self.mha(x_norm)
        x = x + att_out

        # 2) LayerNorm → FeedForward → residual
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x, att_weights, att_scores