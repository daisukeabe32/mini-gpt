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

        # num_heads 個の SelfAttention ヘッド
        self.heads = nn.ModuleList([
            SelfAttention(d_model, d_k) for _ in range(num_heads)
        ])

        # Concat した後、形を d_model に戻す線形変換
        self.W_O = nn.Linear(num_heads * d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)
        戻り値:
          out: (B, T, d_model)
          weights: (B, num_heads, T, T)
          scores: (B, num_heads, T, T)
        """
        B, T, D = x.shape
        assert D == self.d_model

        head_outputs = []   # 各 head の out: (B, T, d_model)
        all_scores = []     # 各 head の scores: (B, T, T)
        all_weights = []    # 各 head の weights: (B, T, T)

        
        for head in self.heads:
            out_h, weights_h, scores_h = head(x)
            head_outputs.append(out_h)
            all_scores.append(scores_h)
            all_weights.append(weights_h)
        print("self.heads", self.heads)
        print("head_outputs", head_outputs)
        print("all_scores", all_scores)
        print("all_weights", all_weights)
        
        # (B, T, num_heads * d_model)
        concat = torch.cat(head_outputs, dim=-1)
        print("concat:", concat)

        # stack して head 次元を明示的に持たせる: (B, num_heads, T, T)
        scores = torch.stack(all_scores, dim=1)
        weights = torch.stack(all_weights, dim=1)

        # (B, T, d_model)
        out = self.W_O(concat)

        return out, weights, scores