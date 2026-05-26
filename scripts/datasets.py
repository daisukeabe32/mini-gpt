"""
Dataset registry for mini-gpt experiments.

Each entry defines how to download and reference a corpus.  Add new datasets
here; download_dataset.py consumes this registry and handles the rest.

Usage (from other scripts):
    from scripts.datasets import REGISTRY, get_config
    cfg = get_config("wikitext103")
    print(cfg["tokenized_subdir"])   # → "bpe_hf_30000_wikitext103"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, dict] = {
    # ------------------------------------------------------------------
    # TinyStories — short children's stories, simple vocabulary
    # Complexity: LOW
    # ------------------------------------------------------------------
    "tinystories": {
        "description": "Short children's stories (TinyStories)",
        "hf_id":        "roneneldan/TinyStories",
        "hf_config":    None,
        "hf_split":     "train",
        "text_field":   "text",
        "separator":    "\n\n",      # written between documents in the .txt file
        "out_path":     "data/tinystories.txt",
        "corpus_name":  "tinystories",
        "tokenized_subdir": "bpe_hf_30000_tinystories",
        "bpe_vocab_size":   30_000,
        "streaming":    True,
        "sample_mb":    200,         # MB of text used for BPE vocab training (None = full)
        "target_mb":    None,        # stop downloading after this many MB (None = all)
    },

    # ------------------------------------------------------------------
    # WikiText-103 — Wikipedia featured articles
    # Complexity: MEDIUM-HIGH
    # ------------------------------------------------------------------
    "wikitext103": {
        "description": "Wikipedia featured articles (WikiText-103)",
        "hf_id":        "Salesforce/wikitext",
        "hf_config":    "wikitext-103-raw-v1",
        "hf_split":     "train",
        "text_field":   "text",
        "separator":    "\n",
        "out_path":     "data/wikitext103.txt",
        "corpus_name":  "wikitext103",
        "tokenized_subdir": "bpe_hf_30000_wikitext103",
        "bpe_vocab_size":   30_000,
        "streaming":    False,       # small enough to load fully into memory
        "sample_mb":    None,        # use full corpus for BPE training
        "target_mb":    None,
    },

    # ------------------------------------------------------------------
    # GitHub Python — Python source code
    # Complexity: STRUCTURAL / HIGH REPETITION
    # ------------------------------------------------------------------
    "github_python": {
        "description": "Python source code from GitHub",
        "hf_id":        "codeparrot/github-code",
        "hf_config":    None,
        "hf_split":     "train",
        "text_field":   "code",
        "separator":    "\n\n",
        "out_path":     "data/github_python.txt",
        "corpus_name":  "github_python",
        "tokenized_subdir": "bpe_hf_30000_github_python",
        "bpe_vocab_size":   30_000,
        "streaming":    True,
        "sample_mb":    200,
        "target_mb":    2_000,       # cap at 2 GB (full dataset is hundreds of GB)
        "hf_filter":    {"programming_language": "Python"},  # column → value filter
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_config(name: str) -> dict:
    """Return the registry entry for *name*, raising KeyError with a hint on miss."""
    if name not in REGISTRY:
        valid = ", ".join(REGISTRY)
        raise KeyError(f"Unknown dataset '{name}'. Valid options: {valid}")
    return REGISTRY[name]


def list_datasets() -> None:
    """Print a summary of all registered datasets."""
    print(f"{'Name':<16}  {'Description'}")
    print("-" * 60)
    for name, cfg in REGISTRY.items():
        print(f"{name:<16}  {cfg['description']}")
