import torch
import pytest
from src.attention import SelfAttention
from src.mha import MultiHeadAttention


B, T, d_model, d_k, num_heads = 2, 6, 16, 4, 4


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(B, T, d_model)


class TestSelfAttention:
    def test_output_shape(self, x):
        sa = SelfAttention(d_model=d_model, d_k=d_k)
        out, weights, scores = sa(x)
        assert out.shape == (B, T, d_k)
        assert weights.shape == (B, T, T)
        assert scores.shape == (B, T, T)

    def test_causal_mask_upper_tri_zero(self, x):
        sa = SelfAttention(d_model=d_model, d_k=d_k)
        _, weights, _ = sa(x)
        # Upper-triangular (excluding diagonal) must be exactly 0
        for b in range(B):
            upper = torch.triu(weights[b], diagonal=1)
            assert upper.abs().max().item() == pytest.approx(0.0)

    def test_attention_weights_sum_to_one(self, x):
        sa = SelfAttention(d_model=d_model, d_k=d_k)
        _, weights, _ = sa(x)
        row_sums = weights.sum(dim=-1)
        assert row_sums.allclose(torch.ones_like(row_sums), atol=1e-5)

    def test_no_nan_or_inf(self, x):
        sa = SelfAttention(d_model=d_model, d_k=d_k)
        out, weights, _ = sa(x)
        assert not torch.isnan(out).any()
        assert not torch.isnan(weights).any()
        assert not torch.isinf(out).any()


class TestMultiHeadAttention:
    def test_output_shape(self, x):
        mha = MultiHeadAttention(d_model=d_model, d_k=d_k, num_heads=num_heads)
        out, weights, scores = mha(x)
        assert out.shape == (B, T, d_model)
        assert weights.shape == (B, num_heads, T, T)
        assert scores.shape == (B, num_heads, T, T)

    def test_causal_mask_upper_tri_zero(self, x):
        mha = MultiHeadAttention(d_model=d_model, d_k=d_k, num_heads=num_heads)
        _, weights, _ = mha(x)
        upper = torch.triu(weights, diagonal=1)
        assert upper.abs().max().item() == pytest.approx(0.0)

    def test_attention_weights_sum_to_one(self, x):
        mha = MultiHeadAttention(d_model=d_model, d_k=d_k, num_heads=num_heads)
        _, weights, _ = mha(x)
        row_sums = weights.sum(dim=-1)
        assert row_sums.allclose(torch.ones_like(row_sums), atol=1e-5)

    def test_no_nan_or_inf(self, x):
        mha = MultiHeadAttention(d_model=d_model, d_k=d_k, num_heads=num_heads)
        out, weights, _ = mha(x)
        assert not torch.isnan(out).any()
        assert not torch.isnan(weights).any()
        assert not torch.isinf(out).any()
