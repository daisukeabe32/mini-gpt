import torch
import pytest
from src.block import TransformerBlock


B, T, d_model, d_k, num_heads, d_ff = 2, 6, 16, 4, 4, 64


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(B, T, d_model)


class TestTransformerBlock:
    def test_output_shape(self, x):
        block = TransformerBlock(d_model=d_model, d_k=d_k, num_heads=num_heads, d_ff=d_ff)
        out, weights, scores = block(x)
        assert out.shape == (B, T, d_model)
        assert weights.shape == (B, num_heads, T, T)
        assert scores.shape == (B, num_heads, T, T)

    def test_residual_preserves_shape(self, x):
        block = TransformerBlock(d_model=d_model, d_k=d_k, num_heads=num_heads, d_ff=d_ff)
        out, _, _ = block(x)
        assert out.shape == x.shape

    def test_gradient_flows(self, x):
        x_req = x.requires_grad_(True)
        block = TransformerBlock(d_model=d_model, d_k=d_k, num_heads=num_heads, d_ff=d_ff)
        out, _, _ = block(x_req)
        out.sum().backward()
        assert x_req.grad is not None
