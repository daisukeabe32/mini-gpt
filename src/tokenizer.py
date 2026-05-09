# src/tokenizer.py

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

    def __init__(self, text: str, vocab_size: int = 512):
        # ---- 1. character-level seed vocabulary -------------------------
        chars = sorted(set(text))
        self.stoi: dict = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict = {i: ch for i, ch in enumerate(chars)}
        # merges: ordered list of ((id_a, id_b), new_id)
        self.merges: list = []

        # ---- 2. BPE training --------------------------------------------
        ids = [self.stoi[ch] for ch in text]   # working token sequence
        n_merges = max(0, vocab_size - len(chars))

        for _ in range(n_merges):
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