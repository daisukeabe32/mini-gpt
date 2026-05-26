"""
Unified dataset download script.

Downloads a registered corpus to disk as a plain-text file.
Run tokenize_corpus.py separately to encode it.

Usage:
    python -m scripts.download_dataset --dataset tinystories
    python -m scripts.download_dataset --dataset wikitext103
    python -m scripts.download_dataset --dataset github_python
    python -m scripts.download_dataset --list
"""

import argparse
import os
import time

from scripts.datasets import REGISTRY, get_config, list_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a registered corpus to a plain-text file"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name (see --list for options)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available datasets and exit",
    )
    parser.add_argument(
        "--out_path", type=str, default=None,
        help="Override output file path (default: from registry)",
    )
    parser.add_argument(
        "--target_mb", type=float, default=None,
        help="Stop after downloading this many MB of text (overrides registry default)",
    )
    return parser.parse_args()


def _write_streaming(cfg: dict, out_path: str, target_mb: float | None) -> None:
    """Download a streaming HuggingFace dataset and write to *out_path*."""
    from datasets import load_dataset

    kwargs = dict(split=cfg["hf_split"], streaming=True, trust_remote_code=True)
    if cfg.get("hf_config"):
        kwargs["name"] = cfg["hf_config"]

    print(f"Loading '{cfg['hf_id']}' (streaming)...")
    ds = load_dataset(cfg["hf_id"], **kwargs)

    # apply column-value filter if specified (e.g. programming_language=Python)
    if cfg.get("hf_filter"):
        for col, val in cfg["hf_filter"].items():
            ds = ds.filter(lambda ex, _c=col, _v=val: ex[_c] == _v)

    text_field = cfg["text_field"]
    sep        = cfg.get("separator", "\n\n")
    t0         = time.time()
    n          = 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex[text_field]
            if not text or not text.strip():
                continue
            f.write(text.strip() + sep)
            n += 1

            if n % 50_000 == 0:
                size_mb = os.path.getsize(out_path) / 1024 / 1024
                elapsed = time.time() - t0
                print(f"  {n:>8,} docs  {size_mb:>7.1f} MB  ({elapsed:.0f}s)", flush=True)

            if target_mb and os.path.getsize(out_path) / 1024 / 1024 >= target_mb:
                print(f"  target_mb={target_mb} MB reached — stopping early")
                break

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Done: {n:,} docs  →  {out_path}  ({size_mb:.0f} MB)  ({time.time()-t0:.0f}s)")


def _write_full(cfg: dict, out_path: str) -> None:
    """Download a non-streaming HuggingFace dataset and write to *out_path*."""
    from datasets import load_dataset

    kwargs = dict(split=cfg["hf_split"])
    if cfg.get("hf_config"):
        kwargs["name"] = cfg["hf_config"]

    print(f"Loading '{cfg['hf_id']}' (full)...")
    ds = load_dataset(cfg["hf_id"], **kwargs)

    text_field = cfg["text_field"]
    sep        = cfg.get("separator", "\n")
    t0         = time.time()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex[text_field]
            if not text or not text.strip():
                continue
            f.write(text.strip() + sep)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Done: {len(ds):,} docs  →  {out_path}  ({size_mb:.0f} MB)  ({time.time()-t0:.0f}s)")


def main():
    args = parse_args()

    if args.list:
        list_datasets()
        return

    if not args.dataset:
        print("Error: specify --dataset <name> or use --list to see options.")
        return

    cfg      = get_config(args.dataset)
    out_path = args.out_path or cfg["out_path"]
    target_mb = args.target_mb if args.target_mb is not None else cfg.get("target_mb")

    print(f"Dataset : {args.dataset}  —  {cfg['description']}")
    print(f"Output  : {out_path}")

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"{out_path} already exists ({size_mb:.0f} MB) — skipping download.")
        print("Delete the file manually if you want to re-download.")
        return

    if cfg["streaming"]:
        _write_streaming(cfg, out_path, target_mb)
    else:
        _write_full(cfg, out_path)


if __name__ == "__main__":
    main()
