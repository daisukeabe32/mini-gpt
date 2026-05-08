"""
Attention analysis script for Option A: Induction Head Detection.

Induction heads are attention heads that implement a "copy" operation:
given a repeated sequence [t1 t2 ... tN t1 t2 ... tN], an induction head
at position N+k attends strongly back to position k (the token that
previously followed the current token's predecessor). This shows up as
a stripe pattern offset by one position below the main diagonal.

Usage:
    python -m scripts.analyze_attention
    python -m scripts.analyze_attention --checkpoint checkpoints/final.pt
    python -m scripts.analyze_attention --seq_len 64 --out_dir figs/
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

from src.model import MiniGPT
from src.tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze attention patterns for induction heads")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--seq_len",    type=int, default=64,
                        help="Half-length of the repeated sequence (total = 2 * seq_len)")
    parser.add_argument("--out_dir",    type=str, default="figs",
                        help="Directory to save figures")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str):
    """Load model and tokenizer from a saved checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]

    model = MiniGPT(
        vocab_size = cfg["vocab_size"],
        d_model    = cfg["d_model"],
        n_layers   = cfg["n_layers"],
        num_heads  = cfg["num_heads"],
        d_k        = cfg["d_k"],
        d_ff       = cfg["d_ff"],
        block_size = cfg["block_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tok = CharTokenizer.__new__(CharTokenizer)
    tok.stoi     = ckpt["stoi"]
    tok.itos     = ckpt["itos"]
    tok.vocab_size = len(tok.stoi)

    print(f"Loaded checkpoint from '{checkpoint_path}'")
    print(f"  step={ckpt['step']} | val_loss={ckpt['val_loss']:.4f}")
    print(f"  d_model={cfg['d_model']} | n_layers={cfg['n_layers']} | num_heads={cfg['num_heads']}")
    return model, tok, cfg


# ---------------------------------------------------------------------------
# Repeated-sequence test input
# ---------------------------------------------------------------------------

def make_repeated_sequence(tok, seq_len: int, device: str, seed: int = 0):
    """
    Build a repeated random token sequence [t0..tN t0..tN] of length 2*seq_len.
    An induction head should attend from position N+k back to position k.
    """
    rng   = np.random.default_rng(seed)
    chars = list(tok.stoi.keys())
    half  = rng.choice(chars, size=seq_len, replace=True).tolist()
    seq   = half + half  # repeat once
    ids   = torch.tensor([tok.stoi[c] for c in seq], dtype=torch.long, device=device)
    return ids.unsqueeze(0), seq  # (1, 2*seq_len)


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention(model, ids):
    """
    Forward pass and collect (att_weights, att_scores) from every block.

    Returns:
        all_weights: list of (1, num_heads, T, T) tensors, one per block
    """
    B, T = ids.shape
    pos  = torch.arange(T, device=ids.device)
    x    = model.token_embedding(ids) + model.pos_embedding(pos)[None]

    all_weights = []
    for block in model.blocks:
        x_norm             = block.ln1(x)
        att_out, weights, _ = block.mha(x_norm)
        all_weights.append(weights.cpu())          # (1, H, T, T)
        x = x + att_out
        x = x + block.ffn(block.ln2(x))

    return all_weights                             # list of n_layers tensors


# ---------------------------------------------------------------------------
# Induction score
# ---------------------------------------------------------------------------

def induction_score(weights: torch.Tensor, seq_len: int) -> float:
    """
    Measure how strongly a single head (T, T) attends to the
    'one-back shifted' diagonal in the repeated-sequence region.

    A perfect induction head has weights[N+k, k] = 1 for k in [1, seq_len-1].
    Score = mean of those off-diagonal entries.
    """
    w  = weights.squeeze(0)   # (T, T)
    T  = w.shape[0]
    N  = seq_len               # start of the repeated half
    scores = []
    for k in range(1, seq_len):
        row = N + k
        col = k                # one position behind the previous occurrence
        if row < T and col < T:
            scores.append(w[row, col].item())
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_attention_grid(all_weights, seq, n_layers, num_heads, seq_len, out_path):
    """
    Grid of heatmaps: rows = layers, cols = heads.
    Red dashed line marks the boundary between the two halves.
    """
    fig = plt.figure(figsize=(3 * num_heads, 3 * n_layers))
    gs  = gridspec.GridSpec(n_layers, num_heads, figure=fig,
                            hspace=0.4, wspace=0.3)

    T = len(seq)
    for layer_idx, weights in enumerate(all_weights):      # weights: (1, H, T, T)
        for head_idx in range(num_heads):
            ax  = fig.add_subplot(gs[layer_idx, head_idx])
            w   = weights[0, head_idx].numpy()             # (T, T)
            sc  = induction_score(weights[:, head_idx], seq_len)

            ax.imshow(w, aspect="auto", cmap="Blues", vmin=0, vmax=w.max())
            ax.axvline(seq_len - 0.5, color="red",  lw=0.8, linestyle="--")
            ax.axhline(seq_len - 0.5, color="red",  lw=0.8, linestyle="--")
            ax.set_title(f"L{layer_idx+1}H{head_idx+1}\n"
                         f"score={sc:.3f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Attention maps — repeated sequence test\n"
                 "(red line = boundary between first and second half)\n"
                 "High induction score → stripe pattern in bottom-left quadrant",
                 fontsize=9)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention grid → {out_path}")


def print_induction_scores(all_weights, n_layers, num_heads, seq_len):
    """Print a ranked table of induction scores for every head."""
    rows = []
    for layer_idx, weights in enumerate(all_weights):
        for head_idx in range(num_heads):
            sc = induction_score(weights[:, head_idx], seq_len)
            rows.append((sc, layer_idx + 1, head_idx + 1))

    rows.sort(reverse=True)
    print("\n=== Induction scores (ranked) ===")
    print(f"{'Rank':>4}  {'Layer':>5}  {'Head':>4}  {'Score':>7}")
    print("-" * 28)
    for rank, (sc, layer, head) in enumerate(rows, 1):
        flag = " ← candidate" if sc > 0.1 else ""
        print(f"{rank:>4}  {layer:>5}  {head:>4}  {sc:>7.4f}{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    model, tok, cfg = load_model(args.checkpoint, device)

    # Clip seq_len to model's block_size limit
    max_half = cfg["block_size"] // 2
    seq_len  = min(args.seq_len, max_half)
    print(f"\nSequence: {seq_len} tokens × 2 = {2 * seq_len} total")

    ids, seq = make_repeated_sequence(tok, seq_len, device)
    all_weights = extract_attention(model, ids)

    # Scores
    print_induction_scores(all_weights, cfg["n_layers"], cfg["num_heads"], seq_len)

    # Heatmap grid
    out_path = os.path.join(args.out_dir, "attention_grid.png")
    plot_attention_grid(
        all_weights, seq,
        cfg["n_layers"], cfg["num_heads"], seq_len,
        out_path,
    )


if __name__ == "__main__":
    main()
