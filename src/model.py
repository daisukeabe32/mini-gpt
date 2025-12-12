# src/model.py

import torch
import torch.nn as nn

from src.block import TransformerBlock


class MiniGPT(nn.Module):
    """
    A tiny GPT-like language model.

    Args:
        vocab_size: size of the vocabulary
        d_model:    embedding dimension
        n_layers:   number of Transformer blocks
        num_heads:  number of attention heads in each block
        d_k:        dimension of each attention head
        d_ff:       hidden size of the feed-forward network
        block_size: maximum context length (T)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        num_heads: int,
        d_k: int,
        d_ff: int,
        block_size: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_ff = d_ff
        self.block_size = block_size

        # 1) token embedding: (B, T) of token ids -> (B, T, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2) position embedding: positions [0 .. block_size-1] -> (T, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)

        # 3) stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, d_k=d_k, num_heads=num_heads, d_ff=d_ff)
            for _ in range(n_layers)
        ])

        # 4) final LayerNorm before output head
        self.ln_f = nn.LayerNorm(d_model)

        # 5) language-model head: (B, T, d_model) -> (B, T, vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) tensor of token ids

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        # Ensure we do not exceed the context window
        assert T <= self.block_size

        # (T,) positions 0..T-1 on the same device as idx
        pos = torch.arange(T, device=idx.device)
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)          # (B, T, d_model)
        pos_emb = self.pos_embedding(pos)[None, :, :]  # (1, T, d_model)

        # Broadcast add: (B, T, d_model)
        x = tok_emb + pos_emb
        
        # Pass through each Transformer block
        for block in self.blocks:
            x, _, _ = block(x)   # ignore attention weights/scores for now

        # Final LayerNorm + linear head to logits
        x = self.ln_f(x)                # (B, T, d_model)
        logits = self.lm_head(x)        # (B, T, vocab_size)
        return logits