import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Map token ids (B, T) to embeddings (B, T, d_model).
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T)  int64 token ids
        returns: (B, T, d_model) float32 embeddings
        """
        return self.embedding(x)