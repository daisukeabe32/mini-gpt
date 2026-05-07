import torch
import pytest
from src.feedforward import FeedForward


B, T, d_model, d_ff = 2, 6, 16, 64


@pytest.fixture
def x():
    torch.manual_seed(4)
    return torch.randn(B, T, d_model)


class TestFeedForward:
    def test_output_shape(self, x):
        ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        out = ffn(x)
        assert out.shape == (B, T, d_model)

    def test_no_nan_or_inf(self, x):
        ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        out = ffn(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flows(self, x):
        x_req = x.requires_grad_(True)
        ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        out = ffn(x_req)
        out.sum().backward()
        assert x_req.grad is not None
