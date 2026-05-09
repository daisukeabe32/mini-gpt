# src/tokenizer.py

import json
import os
from collections import Counter


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer trained from scratch on raw text.

    Algorithm:
        1. Initialise vocabulary with all unique characters in the corpus.
        2. Repeat until vocab reaches target size:
             a. Count every adjacent pair of token IDs in the current encoding.
             b. Merge the most-frequent pair into a new token.
             c. Record the merge; re-encode the corpus with the new token.

    The learned merges are stored in self.merges so that encode() can replay
    them in the same order on any new text.

    Interface is intentionally identical to CharTokenizer so the two can be
    swapped with a single flag in the training script.
    """

    def __init__(self, text: str, vocab_size: int = 512, verbose: bool = False):
        # ---- 1. character-level seed vocabulary -------------------------
        chars = sorted(set(text))
        self.stoi: dict = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict = {i: ch for i, ch in enumerate(chars)}
        # merges: ordered list of ((id_a, id_b), new_id)
        self.merges: list = []

        # ---- 2. BPE training --------------------------------------------
        ids = [self.stoi[ch] for ch in text]   # working token sequence
        n_merges = max(0, vocab_size - len(chars))

        for step in range(n_merges):
            counts = Counter(zip(ids[:-1], ids[1:]))
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            new_id = len(self.stoi)
            new_token = self.itos[best_pair[0]] + self.itos[best_pair[1]]
            self.stoi[new_token] = new_id
            self.itos[new_id] = new_token
            self.merges.append((best_pair, new_id))
            ids = self._apply_merge(ids, best_pair, new_id)
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{n_merges} | vocab={len(self.stoi)} | "
                      f"seq_len={len(ids):,} | last: '{new_token}'")

        self.vocab_size = len(self.stoi)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_merge(self, ids: list, pair: tuple, new_id: int) -> list:
        """Replace every non-overlapping occurrence of pair with new_id."""
        result = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    # ------------------------------------------------------------------
    # Public API  (same as CharTokenizer)
    # ------------------------------------------------------------------

    def encode(self, s: str) -> list:
        """
        Convert a string to a list of BPE token IDs.
        Merges are applied in the training order.
        Raises KeyError for characters not seen during training.
        """
        ids = [self.stoi[ch] for ch in s]
        for pair, new_id in self.merges:
            ids = self._apply_merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list) -> str:
        """Convert a list of BPE token IDs back to the original string."""
        return "".join(self.itos[i] for i in ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: str) -> None:
        """
        Save tokenizer state to dir_path/tokenizer.json.
        Creates the directory if it does not exist.
        """
        os.makedirs(dir_path, exist_ok=True)
        payload = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            # itos keys must be strings in JSON
            "itos": {str(k): v for k, v in self.itos.items()},
            # merges: [[id_a, id_b, new_id], ...]
            "merges": [[p[0], p[1], new_id] for p, new_id in self.merges],
        }
        with open(os.path.join(dir_path, "tokenizer.json"), "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "BPETokenizer":
        """Load a previously saved BPETokenizer from dir_path/tokenizer.json."""
        with open(os.path.join(dir_path, "tokenizer.json")) as f:
            payload = json.load(f)
        assert payload["type"] == "bpe", "tokenizer.json is not a BPE tokenizer"
        tok = cls.__new__(cls)
        tok.itos = {int(k): v for k, v in payload["itos"].items()}
        tok.stoi = {v: int(k) for k, v in payload["itos"].items()}
        tok.merges = [((a, b), new_id) for a, b, new_id in payload["merges"]]
        tok.vocab_size = payload["vocab_size"]
        return tok


class CharTokenizer:
    """
    A simple character-level tokenizer.
    Converts text <-> list of integer token IDs.
    """

    def __init__(self, text: str):
        # Collect all unique characters
        chars = sorted(list(set(text)))
        self.vocab = chars
        self.vocab_size = len(chars)

        # str -> id
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # id -> str
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        """
        Convert string to list of token IDs.
        """
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        """
        Convert list of token IDs back to string.
        """
        return "".join(self.itos[i] for i in ids)

    def save(self, dir_path: str) -> None:
        """Save tokenizer state to dir_path/tokenizer.json."""
        os.makedirs(dir_path, exist_ok=True)
        payload = {
            "type": "char",
            "vocab_size": self.vocab_size,
            "itos": {str(k): v for k, v in self.itos.items()},
            "merges": [],
        }
        with open(os.path.join(dir_path, "tokenizer.json"), "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "CharTokenizer":
        """Load a previously saved CharTokenizer from dir_path/tokenizer.json."""
        with open(os.path.join(dir_path, "tokenizer.json")) as f:
            payload = json.load(f)
        assert payload["type"] == "char", "tokenizer.json is not a char tokenizer"
        tok = cls.__new__(cls)
        tok.itos = {int(k): v for k, v in payload["itos"].items()}
        tok.stoi = {v: int(k) for k, v in payload["itos"].items()}
        tok.vocab = list(tok.stoi.keys())
        tok.vocab_size = payload["vocab_size"]
        return tok