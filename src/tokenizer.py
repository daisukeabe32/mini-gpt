# src/tokenizer.py

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