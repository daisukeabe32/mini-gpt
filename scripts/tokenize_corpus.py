"""
One-time corpus tokenization script.

Trains a tokenizer on a raw text corpus, encodes the full corpus,
and saves everything to data/tokenized/{type}_{vocab_size}_{corpus}/:

    tokenizer.json   — tokenizer vocabulary and BPE merges (human-readable)
    train_ids.pt     — encoded training split (first 90%)
    val_ids.pt       — encoded validation split (last 10%)

Directory naming convention:
    BPE  →  bpe_{vocab_size}_{corpus}    e.g. bpe_3000_shakespeare
    char →  char_{corpus}                e.g. char_shakespeare

If the target directory already exists the script aborts without
overwriting, so previously cached results are always safe.

Usage:
    python -m scripts.tokenize_corpus --tokenizer bpe --bpe_vocab_size 3000
    python -m scripts.tokenize_corpus --tokenizer bpe --bpe_vocab_size 3000 --data_path data/shakespeare.txt
    python -m scripts.tokenize_corpus --tokenizer char
"""

import argparse
import os
import time

import torch

from src.tokenizer import BPETokenizer, CharTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize a text corpus and cache to disk")
    parser.add_argument("--tokenizer",      type=str, default="char",
                        choices=["char", "bpe"])
    parser.add_argument("--bpe_vocab_size", type=int, default=512,
                        help="Target BPE vocabulary size (ignored for char)")
    parser.add_argument("--data_path",      type=str, default="data/shakespeare.txt")
    parser.add_argument("--corpus_name",    type=str, default=None,
                        help="Short corpus label used in the output directory name. "
                             "Defaults to the stem of --data_path (e.g. 'shakespeare').")
    parser.add_argument("--out_dir",        type=str, default="data/tokenized",
                        help="Parent directory for tokenized outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- derive corpus label ------------------------------------------------
    corpus_name = args.corpus_name or os.path.splitext(os.path.basename(args.data_path))[0]

    # ---- output directory ---------------------------------------------------
    if args.tokenizer == "bpe":
        run_name = f"bpe_{args.bpe_vocab_size}_{corpus_name}"
    else:
        run_name = f"char_{corpus_name}"
    out_dir = os.path.join(args.out_dir, run_name)

    if os.path.exists(out_dir):
        print(f"[tokenize_corpus] '{out_dir}' already exists — skipping to avoid overwrite.")
        print("  Delete the directory manually if you want to re-tokenize.")
        return

    # ---- load corpus --------------------------------------------------------
    with open(args.data_path, "r") as f:
        text = f.read()
    print(f"Corpus: {len(text):,} characters")

    # ---- train tokenizer ----------------------------------------------------
    t0 = time.time()
    if args.tokenizer == "bpe":
        print(f"Training BPE tokenizer (vocab_size={args.bpe_vocab_size})...")
        tok = BPETokenizer(text, vocab_size=args.bpe_vocab_size, verbose=True)
    else:
        print("Building char tokenizer...")
        tok = CharTokenizer(text)
    print(f"Tokenizer ready: vocab_size={tok.vocab_size}  ({time.time()-t0:.1f}s)")

    # ---- encode full corpus -------------------------------------------------
    print("Encoding corpus...")
    t1 = time.time()
    ids = tok.encode(text)
    print(f"Encoded: {len(ids):,} tokens  ({time.time()-t1:.1f}s)  "
          f"compression={len(ids)/len(text):.3f}x")

    # ---- 90 / 10 split ------------------------------------------------------
    n = int(0.9 * len(ids))
    train_ids = torch.tensor(ids[:n], dtype=torch.long)
    val_ids   = torch.tensor(ids[n:], dtype=torch.long)
    print(f"Split: train={len(train_ids):,}  val={len(val_ids):,}")

    # ---- save ---------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    tok.save(out_dir)
    torch.save(train_ids, os.path.join(out_dir, "train_ids.pt"))
    torch.save(val_ids,   os.path.join(out_dir, "val_ids.pt"))

    print(f"\nSaved to '{out_dir}':")
    for fname in ["tokenizer.json", "train_ids.pt", "val_ids.pt"]:
        size = os.path.getsize(os.path.join(out_dir, fname))
        print(f"  {fname:20s}  {size/1024:.1f} KB")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
