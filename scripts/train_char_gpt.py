import torch
import torch.nn.functional as F
import wandb

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


@torch.no_grad()
def estimate_loss(model, data_ids, block_size, batch_size, device, eval_iters=50):
    """Estimate loss on a dataset split over multiple batches."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch(data_ids, block_size, batch_size, device)
        logits = model(xb)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), yb.view(B * T))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    torch.manual_seed(42)

    # --------------------------------------------------
    # 1. corpus & tokenizer
    # --------------------------------------------------
    with open("data/shakespeare.txt", "r") as f:
        text = f.read()

    tok = CharTokenizer(text)
    vocab_size = tok.vocab_size

    data_ids = torch.tensor(tok.encode(text), dtype=torch.long)

    # 90/10 train/val split
    n = int(0.9 * len(data_ids))
    train_data = data_ids[:n]
    val_data   = data_ids[n:]

    # --------------------------------------------------
    # 2. hyperparameters
    # --------------------------------------------------
    config = dict(
        block_size  = 128,
        d_model     = 64,
        n_layers    = 4,
        num_heads   = 4,
        d_k         = 16,
        d_ff        = 256,
        batch_size  = 32,
        max_iters   = 5000,
        lr          = 1e-3,
        eval_every  = 200,
        vocab_size  = vocab_size,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # --------------------------------------------------
    # 3. W&B init
    # --------------------------------------------------
    wandb.init(
        project = "mini-gpt",
        entity  = "daisukeabe32",
        config  = config,
    )
    print(f"vocab_size: {vocab_size} | device: {device}")
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    # --------------------------------------------------
    # 4. model
    # --------------------------------------------------
    model = MiniGPT(
        vocab_size = vocab_size,
        d_model    = config["d_model"],
        n_layers   = config["n_layers"],
        num_heads  = config["num_heads"],
        d_k        = config["d_k"],
        d_ff       = config["d_ff"],
        block_size = config["block_size"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # --------------------------------------------------
    # 5. training loop
    # --------------------------------------------------
    model.train()
    for step in range(config["max_iters"]):
        xb, yb = get_batch(train_data, config["block_size"], config["batch_size"], device)
        logits = model(xb)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), yb.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config["eval_every"] == 0 or step == config["max_iters"] - 1:
            train_loss = estimate_loss(model, train_data, config["block_size"], config["batch_size"], device)
            val_loss   = estimate_loss(model, val_data,   config["block_size"], config["batch_size"], device)
            train_bpc  = train_loss / 0.693147  # nats -> bits per char
            val_bpc    = val_loss   / 0.693147
            print(f"step {step:04d} | train loss {train_loss:.4f} ({train_bpc:.3f} bpc) | val loss {val_loss:.4f} ({val_bpc:.3f} bpc)")
            wandb.log({
                "train/loss": train_loss,
                "train/bpc":  train_bpc,
                "val/loss":   val_loss,
                "val/bpc":    val_bpc,
                "step":       step,
            })

    # --------------------------------------------------
    # 6. text generation sample
    # --------------------------------------------------
    model.eval()
    context = torch.tensor([[tok.stoi["h"]]], dtype=torch.long, device=device)
    generated = context
    for _ in range(500):
        idx_cond    = generated[:, -config["block_size"]:]
        logits      = model(idx_cond)
        logits_last = logits[:, -1, :]
        probs       = F.softmax(logits_last, dim=-1)
        next_id     = torch.multinomial(probs, num_samples=1)
        generated   = torch.cat([generated, next_id], dim=1)

    sample_text = tok.decode(generated[0].tolist())
    print("\n=== sample ===")
    print(sample_text)
    wandb.log({"sample_text": wandb.Html(f"<pre>{sample_text}</pre>")})

    wandb.finish()


if __name__ == "__main__":
    main()
