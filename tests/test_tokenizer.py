import os
import tempfile

import pytest
from src.tokenizer import BPETokenizer, CharTokenizer, HFBPETokenizer


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


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------

# Corpus where "ab" is the dominant pair — easy to reason about merges.
BPE_CORPUS = "ababababab cd cd cd"


@pytest.fixture
def bpe_tok():
    # Request one merge beyond the char vocabulary.
    chars = len(set(BPE_CORPUS))
    return BPETokenizer(BPE_CORPUS, vocab_size=chars + 1)


@pytest.fixture
def bpe_tok_full():
    # Request enough merges to cover common digrams (used for roundtrip tests).
    return BPETokenizer(BPE_CORPUS, vocab_size=len(set(BPE_CORPUS)) + 10)


class TestBPETokenizer:
    def test_vocab_size_matches_request(self, bpe_tok):
        chars = len(set(BPE_CORPUS))
        assert bpe_tok.vocab_size == chars + 1

    def test_vocab_size_does_not_exceed_request(self):
        # When the corpus is tiny, merges stop early rather than crashing.
        tok = BPETokenizer("aab", vocab_size=100)
        assert tok.vocab_size <= 100

    def test_first_merge_is_most_frequent_pair(self, bpe_tok):
        # In BPE_CORPUS "ab" appears 5 times — must be the first merge.
        assert bpe_tok.merges[0][0] == (bpe_tok.stoi["a"], bpe_tok.stoi["b"])

    def test_merged_token_in_vocab(self, bpe_tok):
        # After merging "a"+"b", the token "ab" should be in stoi.
        assert "ab" in bpe_tok.stoi

    def test_encode_applies_merge(self, bpe_tok):
        # "ab" should encode to a single ID (the merged token), not two IDs.
        ids = bpe_tok.encode("ab")
        assert ids == [bpe_tok.stoi["ab"]]

    def test_encode_decode_roundtrip(self, bpe_tok_full):
        assert bpe_tok_full.decode(bpe_tok_full.encode(BPE_CORPUS)) == BPE_CORPUS

    def test_encode_returns_ints(self, bpe_tok):
        ids = bpe_tok.encode("ab")
        assert all(isinstance(i, int) for i in ids)

    def test_compression_ratio(self, bpe_tok_full):
        # BPE-encoded sequence must be strictly shorter than char-encoded.
        char_len = len(BPE_CORPUS)
        bpe_len = len(bpe_tok_full.encode(BPE_CORPUS))
        assert bpe_len < char_len

    def test_stoi_itos_consistency(self, bpe_tok_full):
        for token, idx in bpe_tok_full.stoi.items():
            assert bpe_tok_full.itos[idx] == token

    def test_unknown_char_raises(self, bpe_tok):
        with pytest.raises(KeyError):
            bpe_tok.encode("Z")


# ---------------------------------------------------------------------------
# HFBPETokenizer
# ---------------------------------------------------------------------------

HF_CORPUS = (
    "Once upon a time there was a small cat. "
    "The cat sat on the mat. "
    "Once upon a time there was a small dog. "
    "The dog ran in the park. " * 20
)


@pytest.fixture
def hf_tok_file(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(HF_CORPUS, encoding="utf-8")
    return str(p)


@pytest.fixture
def hf_tok(hf_tok_file):
    chars = len(set(HF_CORPUS))
    return HFBPETokenizer(hf_tok_file, vocab_size=chars + 10)


class TestHFBPETokenizer:
    def test_vocab_size_set(self, hf_tok):
        assert hf_tok.vocab_size > 0

    def test_encode_returns_ints(self, hf_tok):
        ids = hf_tok.encode("Once upon a time")
        assert ids and all(isinstance(i, int) for i in ids)

    def test_decode_returns_string(self, hf_tok):
        ids = hf_tok.encode("Once upon a time")
        result = hf_tok.decode(ids)
        assert isinstance(result, str) and len(result) > 0

    def test_stoi_itos_consistency(self, hf_tok):
        for token, idx in list(hf_tok.stoi.items())[:50]:
            assert hf_tok.itos[idx] == token

    def test_compression(self, hf_tok, hf_tok_file):
        ids = hf_tok.encode_file(hf_tok_file)
        assert len(ids) < len(HF_CORPUS)

    def test_save_load_roundtrip(self, hf_tok, tmp_path):
        hf_tok.save(str(tmp_path))
        loaded = HFBPETokenizer.load(str(tmp_path))
        assert loaded.vocab_size == hf_tok.vocab_size
        ids_orig   = hf_tok.encode("Once upon a time")
        ids_loaded = loaded.encode("Once upon a time")
        assert ids_orig == ids_loaded

    def test_tokenizer_json_type(self, hf_tok, tmp_path):
        import json
        hf_tok.save(str(tmp_path))
        meta = json.load(open(tmp_path / "tokenizer.json"))
        assert meta["type"] == "bpe_hf"
