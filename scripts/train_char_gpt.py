import torch
import torch.nn.functional as F

from src.tokenizer import CharTokenizer
from src.model import MiniGPT


def get_batch(data_ids: torch.Tensor, block_size: int, batch_size: int, device: str):
    """
    data_ids: (N,)
    returns:
        x: (B, T)
        y: (B, T)
    """
    N = data_ids.size(0)
    ix = torch.randint(0, N - block_size - 1, (batch_size,), device=device)

    x = torch.stack([data_ids[i:i + block_size] for i in ix])
    y = torch.stack([data_ids[i + 1:i + 1 + block_size] for i in ix])

    return x.to(device), y.to(device)


def main():
    # -----------------------------
    # 1. tiny corpus & tokenizer
    # -----------------------------
    text = "hello world!\n" * 200  # 小さなおもちゃデータ

    tok = CharTokenizer(text)
    vocab_size = tok.vocab_size
    print("vocab_size:", vocab_size)

    data_ids = torch.tensor(tok.encode(text), dtype=torch.long)

    # -----------------------------
    # 2. model hyper-parameters
    # -----------------------------
    block_size = 8      # コンテキスト長（最大トークン数）
    d_model = 32
    n_layers = 2
    num_heads = 4
    d_k = 8
    d_ff = 64
    batch_size = 16
    max_iters = 200

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        d_k=d_k,
        d_ff=d_ff,
        block_size=block_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # -----------------------------
    # 3. tiny training loop
    # -----------------------------
    model.train()
    for step in range(max_iters):
        xb, yb = get_batch(data_ids, block_size, batch_size, device)
        logits = model(xb)  # (B, T, vocab_size)
        B, T, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B * T, V),
            yb.view(B * T),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"step {step:03d} | loss {loss.item():.4f}")

    # -----------------------------
    # 4. sampling a little text
    # -----------------------------
    model.eval()
    context = torch.tensor([[tok.stoi["h"]]], dtype=torch.long, device=device)

    generated = context
    for _ in range(50):
        # use only the last `block_size` tokens as context
        idx_cond = generated[:, -block_size:]
        logits = model(idx_cond)          # (B, T, V)
        logits_last = logits[:, -1, :]   # (B, V)
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
        generated = torch.cat([generated, next_id], dim=1)

    out_text = tok.decode(generated[0].tolist())
    print("\n=== sample ===")
    print(out_text)


if __name__ == "__main__":
    main()