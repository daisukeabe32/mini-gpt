import pytest
from src.tokenizer import CharTokenizer


TEXT = "hello world!"


@pytest.fixture
def tok():
    return CharTokenizer(TEXT)


class TestCharTokenizer:
    def test_vocab_size(self, tok):
        assert tok.vocab_size == len(set(TEXT))

    def test_encode_decode_roundtrip(self, tok):
        assert tok.decode(tok.encode(TEXT)) == TEXT

    def test_encode_returns_ints(self, tok):
        ids = tok.encode("hello")
        assert all(isinstance(i, int) for i in ids)

    def test_unknown_char_raises(self, tok):
        with pytest.raises(KeyError):
            tok.encode("Z")
