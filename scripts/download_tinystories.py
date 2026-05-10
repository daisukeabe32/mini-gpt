"""
Download the TinyStories dataset and write it to data/tinystories.txt.

Uses streaming so the full dataset is never loaded into Python memory at once.
Each story is separated by a blank line in the output file.

Usage:
    python -m scripts.download_tinystories
    python -m scripts.download_tinystories --out_path data/tinystories.txt
"""

import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Download TinyStories to a text file")
    parser.add_argument("--out_path", type=str, default="data/tinystories.txt")
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Stop after this many stories (for testing; default: all)")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.out_path):
        size_mb = os.path.getsize(args.out_path) / 1024 / 1024
        print(f"{args.out_path} already exists ({size_mb:.0f} MB) — skipping.")
        return

    from datasets import load_dataset

    print("Loading TinyStories (streaming)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    t0 = time.time()
    n = 0
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            f.write(ex["text"].strip() + "\n\n")
            n += 1
            if n % 100_000 == 0:
                size_mb = os.path.getsize(args.out_path) / 1024 / 1024
                elapsed = time.time() - t0
                print(f"  {n:>7,} stories  {size_mb:>6.0f} MB  ({elapsed:.0f}s)", flush=True)
            if args.max_stories and n >= args.max_stories:
                break

    size_mb = os.path.getsize(args.out_path) / 1024 / 1024
    print(f"Done: {n:,} stories  →  {args.out_path}  ({size_mb:.0f} MB)  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
