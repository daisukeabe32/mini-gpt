import torch
import torch.nn.functional as F
import pytest
from src.tokenizer import CharTokenizer
from src.model import MiniGPT


@pytest.fixture
def tiny_setup():
    torch.manual_seed(42)
    text = "hello world! " * 50
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    model = MiniGPT(
        vocab_size=tok.vocab_size,
        d_model=16, n_layers=2, num_heads=2, d_k=8, d_ff=32, block_size=16,
    )
    return model, data, tok


class TestTraining:
    def test_loss_decreases(self, tiny_setup):
        """Verify that loss decreases after 10 training steps."""
        model, data, tok = tiny_setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        def get_loss():
            x = data[:16].unsqueeze(0)
            y = data[1:17].unsqueeze(0)
            logits = model(x)
            B, T, V = logits.shape
            return F.cross_entropy(logits.view(B * T, V), y.view(B * T))

        loss_before = get_loss().item()

        model.train()
        for _ in range(10):
            x = data[:16].unsqueeze(0)
            y = data[1:17].unsqueeze(0)
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_after = get_loss().item()
        assert loss_after < loss_before

    def test_initial_loss_near_log_vocab(self, tiny_setup):
        """Initial loss should be near log(vocab_size), indicating unbiased weight initialization."""
        model, data, tok = tiny_setup
        x = data[:16].unsqueeze(0)
        y = data[1:17].unsqueeze(0)
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
        import math
        expected = math.log(tok.vocab_size)
        assert abs(loss.item() - expected) < 1.5
