import torch
import pytest
from src.model import MiniGPT


VOCAB_SIZE = 65
D_MODEL = 32
N_LAYERS = 2
NUM_HEADS = 4
D_K = 8
D_FF = 128
BLOCK_SIZE = 16

B, T = 2, 10


@pytest.fixture
def model():
    torch.manual_seed(2)
    return MiniGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        d_k=D_K,
        d_ff=D_FF,
        block_size=BLOCK_SIZE,
    )


@pytest.fixture
def idx():
    torch.manual_seed(3)
    return torch.randint(0, VOCAB_SIZE, (B, T))


class TestMiniGPT:
    def test_logits_shape(self, model, idx):
        logits = model(idx)
        assert logits.shape == (B, T, VOCAB_SIZE)

    def test_context_limit_raises(self, model):
        bad_idx = torch.zeros(1, BLOCK_SIZE + 1, dtype=torch.long)
        with pytest.raises(AssertionError):
            model(bad_idx)

    def test_gradient_flows(self, model, idx):
        logits = model(idx)
        loss = logits.mean()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_no_nan_or_inf(self, model, idx):
        logits = model(idx)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
