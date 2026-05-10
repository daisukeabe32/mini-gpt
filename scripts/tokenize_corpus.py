"""
One-time corpus tokenization script.

Trains a tokenizer on a raw text corpus, encodes the full corpus,
and saves everything to data/tokenized/{type}_{vocab_size}_{corpus}/:

    tokenizer.json   — tokenizer metadata (and HF tokenizer state for bpe_hf)
    train_ids.pt     — encoded training split (first 90%)
    val_ids.pt       — encoded validation split (last 10%)

Directory naming convention:
    bpe_hf →  bpe_hf_{vocab_size}_{corpus}   e.g. bpe_hf_30000_tinystories
    BPE    →  bpe_{vocab_size}_{corpus}       e.g. bpe_3000_shakespeare
    char   →  char_{corpus}                   e.g. char_shakespeare

bpe_hf uses the Rust-backed HuggingFace tokenizers library and is the recommended
choice for corpora larger than ~10 MB. It trains directly from the file on disk
without loading the full text into Python memory.

If the target directory already exists the script aborts without overwriting.

Usage:
    python -m scripts.tokenize_corpus --tokenizer bpe_hf --bpe_vocab_size 30000 \\
        --data_path data/tinystories.txt
    python -m scripts.tokenize_corpus --tokenizer bpe --bpe_vocab_size 3000 \\
        --data_path data/shakespeare.txt
    python -m scripts.tokenize_corpus --tokenizer char
"""

import argparse
import os
import time

import numpy as np
import torch

from src.tokenizer import BPETokenizer, CharTokenizer, HFBPETokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize a text corpus and cache to disk")
    parser.add_argument("--tokenizer",      type=str, default="char",
                        choices=["char", "bpe", "bpe_hf"])
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
    if args.tokenizer == "bpe_hf":
        run_name = f"bpe_hf_{args.bpe_vocab_size}_{corpus_name}"
    elif args.tokenizer == "bpe":
        run_name = f"bpe_{args.bpe_vocab_size}_{corpus_name}"
    else:
        run_name = f"char_{corpus_name}"
    out_dir = os.path.join(args.out_dir, run_name)

    if os.path.exists(out_dir):
        print(f"[tokenize_corpus] '{out_dir}' already exists — skipping to avoid overwrite.")
        print("  Delete the directory manually if you want to re-tokenize.")
        return

    file_size_mb = os.path.getsize(args.data_path) / 1024 / 1024
    print(f"Corpus: {args.data_path}  ({file_size_mb:.1f} MB)")

    t0 = time.time()

    # ---- bpe_hf path: file-based, no full load into memory ------------------
    if args.tokenizer == "bpe_hf":
        print(f"Training HF BPE tokenizer (vocab_size={args.bpe_vocab_size})...")
        tok = HFBPETokenizer(args.data_path, vocab_size=args.bpe_vocab_size, verbose=True)
        print(f"Tokenizer ready: vocab_size={tok.vocab_size}  ({time.time()-t0:.1f}s)")

        print("Encoding corpus (chunked)...")
        t1 = time.time()
        ids_arr = tok.encode_file(args.data_path)
        n_chars = int(file_size_mb * 1024 * 1024)
        print(f"Encoded: {len(ids_arr):,} tokens  ({time.time()-t1:.1f}s)  "
              f"compression={len(ids_arr)/n_chars:.3f}x")

        n = int(0.9 * len(ids_arr))
        train_ids = torch.tensor(ids_arr[:n], dtype=torch.long)
        val_ids   = torch.tensor(ids_arr[n:], dtype=torch.long)

    # ---- bpe / char path: load text into memory -----------------------------
    else:
        with open(args.data_path, "r") as f:
            text = f.read()
        print(f"Corpus characters: {len(text):,}")

        if args.tokenizer == "bpe":
            print(f"Training BPE tokenizer (vocab_size={args.bpe_vocab_size})...")
            tok = BPETokenizer(text, vocab_size=args.bpe_vocab_size, verbose=True)
        else:
            print("Building char tokenizer...")
            tok = CharTokenizer(text)
        print(f"Tokenizer ready: vocab_size={tok.vocab_size}  ({time.time()-t0:.1f}s)")

        print("Encoding corpus...")
        t1 = time.time()
        ids = tok.encode(text)
        print(f"Encoded: {len(ids):,} tokens  ({time.time()-t1:.1f}s)  "
              f"compression={len(ids)/len(text):.3f}x")

        n = int(0.9 * len(ids))
        train_ids = torch.tensor(ids[:n], dtype=torch.long)
        val_ids   = torch.tensor(ids[n:], dtype=torch.long)

    # ---- save ---------------------------------------------------------------
    print(f"Split: train={len(train_ids):,}  val={len(val_ids):,}")
    os.makedirs(out_dir, exist_ok=True)
    tok.save(out_dir)
    torch.save(train_ids, os.path.join(out_dir, "train_ids.pt"))
    torch.save(val_ids,   os.path.join(out_dir, "val_ids.pt"))

    print(f"\nSaved to '{out_dir}':")
    for fname in os.listdir(out_dir):
        size = os.path.getsize(os.path.join(out_dir, fname))
        print(f"  {fname:30s}  {size/1024:.1f} KB")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
