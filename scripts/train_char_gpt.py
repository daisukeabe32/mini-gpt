import argparse
import math
import os
import torch
import torch.nn.functional as F
import wandb

from src.tokenizer import BPETokenizer, CharTokenizer, HFBPETokenizer
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a character-level GPT on TinyShakespeare")
    parser.add_argument("--block_size",  type=int,   default=256)
    parser.add_argument("--d_model",     type=int,   default=384)
    parser.add_argument("--n_layers",    type=int,   default=6)
    parser.add_argument("--num_heads",   type=int,   default=8)
    parser.add_argument("--d_k",         type=int,   default=48)
    parser.add_argument("--d_ff",        type=int,   default=1536)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--max_iters",   type=int,   default=10000)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--eval_every",  type=int,   default=500)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--min_lr",      type=float, default=3e-5,
                        help="Minimum learning rate at end of cosine decay (1/10 of lr)")
    parser.add_argument("--warmup_iters",type=int,   default=200,
                        help="Steps over which lr linearly warms up from 0 to lr")
    parser.add_argument("--grad_clip",    type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled)")
    parser.add_argument("--tokenizer",    type=str,   default="char",
                        choices=["char", "bpe", "bpe_hf"],
                        help="Tokenizer type: 'char', 'bpe', or 'bpe_hf'")
    parser.add_argument("--bpe_vocab_size", type=int, default=512,
                        help="BPE vocabulary size (used only when --tokenizer bpe)")
    parser.add_argument("--tokenized_dir", type=str,  default=None,
                        help="Load pre-tokenized data from this directory "
                             "(skips tokenizer training and encoding)")
    parser.add_argument("--no_wandb",      action="store_true",
                        help="Disable W&B logging (useful for quick test runs)")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable checkpoint saving (useful for quick test runs)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Base directory for checkpoint saving (e.g. a Google Drive path)")
    parser.add_argument("--save_every",     type=int, default=5000,
                        help="Save a numbered checkpoint every N steps for emergence analysis "
                             "(0 = disabled). Snapshots are saved at eval points that are "
                             "multiples of this value.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)

    # --------------------------------------------------
    # 1. corpus & tokenizer
    # --------------------------------------------------
    if args.tokenized_dir:
        # Fast path: load pre-tokenized data from disk
        print(f"Loading pre-tokenized data from '{args.tokenized_dir}'...")
        import json as _json
        with open(os.path.join(args.tokenized_dir, "tokenizer.json")) as _f:
            _meta = _json.load(_f)
        tok_type = _meta["type"]
        if tok_type == "bpe_hf":
            tok = HFBPETokenizer.load(args.tokenized_dir)
        elif tok_type == "bpe":
            tok = BPETokenizer.load(args.tokenized_dir)
        else:
            tok = CharTokenizer.load(args.tokenized_dir)
        train_data = torch.load(os.path.join(args.tokenized_dir, "train_ids.pt"))
        val_data   = torch.load(os.path.join(args.tokenized_dir, "val_ids.pt"))
        print(f"Loaded: vocab_size={tok.vocab_size}  "
              f"train={len(train_data):,}  val={len(val_data):,}")
    else:
        # Standard path: tokenize from raw text
        with open("data/shakespeare.txt", "r") as f:
            text = f.read()
        if args.tokenizer == "bpe":
            print(f"Training BPE tokenizer (vocab_size={args.bpe_vocab_size})...")
            tok = BPETokenizer(text, vocab_size=args.bpe_vocab_size)
            print(f"BPE tokenizer ready: vocab_size={tok.vocab_size}")
        else:
            tok = CharTokenizer(text)
        data_ids = torch.tensor(tok.encode(text), dtype=torch.long)
        n = int(0.9 * len(data_ids))
        train_data = data_ids[:n]
        val_data   = data_ids[n:]
    vocab_size = tok.vocab_size

    # --------------------------------------------------
    # 2. hyperparameters
    # --------------------------------------------------
    config = dict(
        block_size   = args.block_size,
        d_model      = args.d_model,
        n_layers     = args.n_layers,
        num_heads    = args.num_heads,
        d_k          = args.d_k,
        d_ff         = args.d_ff,
        batch_size   = args.batch_size,
        max_iters    = args.max_iters,
        lr           = args.lr,
        eval_every   = args.eval_every,
        dropout      = args.dropout,
        min_lr       = args.min_lr,
        warmup_iters = args.warmup_iters,
        grad_clip    = args.grad_clip,
        tokenizer    = args.tokenizer,
        vocab_size   = vocab_size,
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # --------------------------------------------------
    # 3. W&B init
    # --------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project = "mini-gpt",
            entity  = "daisukeabe32-university-of-tokyo",
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
        dropout    = config["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    if use_wandb:
        wandb.config.update({"n_params": n_params})

    decay_params    = [p for _, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for _, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config["lr"])

    # --------------------------------------------------
    # 5. training loop
    # --------------------------------------------------
    def get_lr(step: int) -> float:
        """Cosine decay with linear warmup.
        - Steps 0..warmup_iters: lr linearly rises from 0 to max lr
        - Steps warmup_iters..max_iters: lr follows cosine curve down to min_lr
        """
        max_lr = config["lr"]
        min_lr = config["min_lr"]
        warmup = config["warmup_iters"]
        total  = config["max_iters"]
        if step < warmup:
            return max_lr * step / warmup
        decay_ratio = (step - warmup) / (total - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    save_ckpt = not args.no_checkpoint
    if save_ckpt:
        from datetime import datetime
        run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(args.checkpoint_dir, run_id)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"checkpoint dir: {ckpt_dir}")
    best_val_loss = float("inf")

    def save_checkpoint(tag: str, step: int, val_loss: float):
        """tag is 'best' or 'final'."""
        if not save_ckpt:
            return
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "config":      config,
            "stoi":        tok.stoi,
            "itos":        tok.itos,
            "step":        step,
            "val_loss":    val_loss,
        }, path)
        return path

    model.train()
    grad_norms = []
    for step in range(config["max_iters"]):
        # Update learning rate according to cosine schedule
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb = get_batch(train_data, config["block_size"], config["batch_size"], device)
        logits = model(xb)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), yb.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"))
        grad_norms.append(grad_norm.item())
        optimizer.step()

        if step % config["eval_every"] == 0 or step == config["max_iters"] - 1:
            train_loss = estimate_loss(model, train_data, config["block_size"], config["batch_size"], device)
            val_loss   = estimate_loss(model, val_data,   config["block_size"], config["batch_size"], device)
            train_bpc  = train_loss / 0.693147  # nats -> bits per char
            val_bpc    = val_loss   / 0.693147
            mean_grad_norm = sum(grad_norms) / len(grad_norms)
            grad_norms = []
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint("best", step, val_loss)
                print(f"  → best checkpoint saved (val_loss={val_loss:.4f})")
            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                save_checkpoint(f"step_{step:06d}", step, val_loss)
                print(f"  → snapshot saved (step={step})")
            print(f"step {step:04d} | train loss {train_loss:.4f} ({train_bpc:.3f} bpc) | val loss {val_loss:.4f} ({val_bpc:.3f} bpc) | grad_norm {mean_grad_norm:.3f}")
            if use_wandb:
                wandb.log({
                    "train/loss":      train_loss,
                    "train/bpc":       train_bpc,
                    "val/loss":        val_loss,
                    "val/bpc":         val_bpc,
                    "train/grad_norm": mean_grad_norm,
                }, step=step)

    save_checkpoint("final", config["max_iters"] - 1, val_loss)
    print("final checkpoint saved")

    # --------------------------------------------------
    # 6. text generation sample
    # --------------------------------------------------
    model.eval()
    start_text = "h" if isinstance(tok, CharTokenizer) else "Once "
    start_ids = tok.encode(start_text)
    context = torch.tensor([start_ids], dtype=torch.long, device=device)
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
    if use_wandb:
        wandb.log({"sample_text": wandb.Html(f"<pre>{sample_text}</pre>")})
        wandb.finish()


if __name__ == "__main__":
    main()
