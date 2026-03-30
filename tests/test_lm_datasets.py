"""Tests for lm_datasets.py."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lm_datasets import (
    encode,
    decode,
    VOCAB_SIZE,
    _TokenDataset,
    _to_loaders,
    LM_DATASETS,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_encode_returns_list(self):
        assert isinstance(encode("hello"), list)

    def test_encode_bytes_in_range(self):
        tokens = encode("hello world")
        assert all(0 <= t <= 255 for t in tokens)

    def test_decode_roundtrip(self):
        s = "The quick brown fox"
        assert decode(encode(s)) == s

    def test_vocab_size_is_256(self):
        assert VOCAB_SIZE == 256

    def test_empty_string(self):
        assert encode("") == []
        assert decode([]) == ""


# ---------------------------------------------------------------------------
# _TokenDataset
# ---------------------------------------------------------------------------

class TestTokenDataset:
    def test_len_is_n_chunks(self):
        arr = np.arange(1000, dtype=np.uint8)
        ds = _TokenDataset(arr, seq_len=64)
        assert len(ds) == 1000 // 64  # 15

    def test_item_shape(self):
        arr = np.arange(1000, dtype=np.uint8)
        ds = _TokenDataset(arr, seq_len=64)
        x = ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.shape == (64,)
        assert x.dtype == torch.int64

    def test_no_overlap_between_chunks(self):
        arr = np.arange(256, dtype=np.uint8)
        ds = _TokenDataset(arr, seq_len=64)
        chunk0 = ds[0].numpy()
        chunk1 = ds[1].numpy()
        # Chunk 1 starts right after chunk 0 ends
        assert chunk0[-1] + 1 == chunk1[0]

    def test_last_incomplete_chunk_dropped(self):
        # 100 tokens, seq_len=64 → 1 complete chunk, 36 dropped
        arr = np.arange(100, dtype=np.uint8)
        ds = _TokenDataset(arr, seq_len=64)
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# _to_loaders
# ---------------------------------------------------------------------------

class TestToLoaders:
    def _make_tokens(self, n=2000):
        return np.random.randint(0, 256, size=n, dtype=np.uint8)

    def test_returns_two_dataloaders(self):
        tokens = self._make_tokens()
        train_l, val_l = _to_loaders(tokens, seq_len=64, batch_size=16,
                                      train_frac=0.8, random_state=0)
        assert isinstance(train_l, DataLoader)
        assert isinstance(val_l,   DataLoader)

    def test_train_larger_than_val(self):
        tokens = self._make_tokens(4000)
        train_l, val_l = _to_loaders(tokens, seq_len=64, batch_size=16,
                                      train_frac=0.8, random_state=0)
        n_train = sum(x.shape[0] for x in train_l)
        n_val   = sum(x.shape[0] for x in val_l)
        assert n_train > n_val

    def test_batch_shape(self):
        tokens = self._make_tokens(2000)
        train_l, _ = _to_loaders(tokens, seq_len=64, batch_size=8,
                                   train_frac=0.8, random_state=0)
        x = next(iter(train_l))
        assert x.ndim == 2
        assert x.shape[1] == 64

    def test_token_values_in_range(self):
        tokens = self._make_tokens(2000)
        train_l, val_l = _to_loaders(tokens, seq_len=64, batch_size=16,
                                       train_frac=0.8, random_state=0)
        for loader in (train_l, val_l):
            for x in loader:
                assert x.min() >= 0
                assert x.max() <= 255

    def test_reproducible_with_same_seed(self):
        tokens = self._make_tokens(2000)
        _, val1 = _to_loaders(tokens, seq_len=64, batch_size=16,
                               train_frac=0.8, random_state=42)
        _, val2 = _to_loaders(tokens, seq_len=64, batch_size=16,
                               train_frac=0.8, random_state=42)
        x1 = next(iter(val1))
        x2 = next(iter(val2))
        assert torch.equal(x1, x2)

    def test_different_seeds_differ(self):
        tokens = self._make_tokens(2000)
        _, val1 = _to_loaders(tokens, seq_len=64, batch_size=16,
                               train_frac=0.8, random_state=0)
        _, val2 = _to_loaders(tokens, seq_len=64, batch_size=16,
                               train_frac=0.8, random_state=99)
        x1 = next(iter(val1))
        x2 = next(iter(val2))
        assert not torch.equal(x1, x2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestLMDatasetsRegistry:
    def test_registry_keys(self):
        assert set(LM_DATASETS.keys()) == {"tinystories", "wikitext103"}

    def test_all_callable(self):
        for k, fn in LM_DATASETS.items():
            assert callable(fn), f"LM_DATASETS['{k}'] is not callable"


# ---------------------------------------------------------------------------
# Tokenizer — additional tests
# ---------------------------------------------------------------------------

class TestTokenizerAdditional:
    def test_encode_length_matches_ascii(self):
        # ASCII characters are single bytes
        s = "hello"
        assert len(encode(s)) == 5

    def test_encode_decode_unicode(self):
        # Multi-byte UTF-8 chars round-trip correctly
        s = "café"
        assert decode(encode(s)) == s

    def test_encode_all_values_are_ints(self):
        tokens = encode("test string 123")
        assert all(isinstance(t, int) for t in tokens)

    def test_decode_list_vs_bytes(self):
        # decode should handle any sequence of ints in 0-255
        tokens = [72, 101, 108, 108, 111]  # "Hello"
        assert decode(tokens) == "Hello"

    def test_newline_roundtrip(self):
        s = "line1\nline2\n"
        assert decode(encode(s)) == s

    def test_encode_space(self):
        tokens = encode(" ")
        assert tokens == [32]  # ASCII space

    def test_decode_zero_byte(self):
        # Null byte should not raise
        result = decode([0])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _TokenDataset — additional tests
# ---------------------------------------------------------------------------

class TestTokenDatasetAdditional:
    def test_exact_multiple_no_remainder(self):
        # 128 tokens, seq_len=64 → exactly 2 chunks, nothing dropped
        arr = np.arange(128, dtype=np.uint8)
        ds = _TokenDataset(arr, seq_len=64)
        assert len(ds) == 2

    def test_all_chunks_correct_length(self):
        arr = np.arange(300, dtype=np.uint8)
        ds = _TokenDataset(arr, seq_len=64)
        for i in range(len(ds)):
            assert ds[i].shape == (64,)

    def test_chunk_values_match_source(self):
        arr = np.arange(256, dtype=np.uint8)
        ds = _TokenDataset(arr, seq_len=64)
        # Chunk 2 should be tokens 128..191
        chunk = ds[2].numpy()
        expected = np.arange(128, 192, dtype=np.int64)
        np.testing.assert_array_equal(chunk, expected)

    def test_seq_len_equals_array_len(self):
        # Exactly one chunk when seq_len == len(arr)
        arr = np.arange(64, dtype=np.uint8)
        ds = _TokenDataset(arr, seq_len=64)
        assert len(ds) == 1

    def test_seq_len_larger_than_array(self):
        # No complete chunks → length 0
        arr = np.arange(32, dtype=np.uint8)
        ds = _TokenDataset(arr, seq_len=64)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# _to_loaders — additional tests
# ---------------------------------------------------------------------------

class TestToLoadersAdditional:
    def _make_tokens(self, n=4000):
        return np.random.randint(0, 256, size=n, dtype=np.uint8)

    def test_train_no_overlap_with_val(self):
        # The union of train + val chunks should cover all n_chunks indices
        # (no chunk is in both splits)
        tokens = np.arange(640, dtype=np.uint8)  # 10 chunks of 64
        train_l, val_l = _to_loaders(tokens, seq_len=64, batch_size=2,
                                      train_frac=0.8, random_state=0)
        n_train = sum(x.shape[0] for x in train_l)
        n_val   = sum(x.shape[0] for x in val_l)
        assert n_train + n_val == 10

    def test_val_loader_not_shuffled(self):
        # Val loader shuffle=False: iterating twice gives the same order
        tokens = self._make_tokens(4000)
        _, val_l = _to_loaders(tokens, seq_len=64, batch_size=8,
                                train_frac=0.8, random_state=0)
        batches1 = [x.clone() for x in val_l]
        batches2 = [x.clone() for x in val_l]
        for b1, b2 in zip(batches1, batches2):
            assert torch.equal(b1, b2)

    def test_train_drop_last(self):
        # Train loader uses drop_last=True → all batches are full size
        tokens = self._make_tokens(4000)
        train_l, _ = _to_loaders(tokens, seq_len=64, batch_size=8,
                                   train_frac=0.8, random_state=0)
        for x in train_l:
            assert x.shape[0] == 8

    def test_dtype_is_int64(self):
        tokens = self._make_tokens()
        train_l, val_l = _to_loaders(tokens, seq_len=64, batch_size=8,
                                      train_frac=0.8, random_state=0)
        x = next(iter(train_l))
        assert x.dtype == torch.int64

    def test_train_frac_one_gives_no_val(self):
        # train_frac=1.0 → all chunks in train; val falls back to first chunk
        tokens = self._make_tokens()
        train_l, val_l = _to_loaders(tokens, seq_len=64, batch_size=8,
                                      train_frac=1.0, random_state=0)
        n_train = sum(x.shape[0] for x in train_l)
        assert n_train > 0
        # val_l uses fallback (1 chunk) — just check it doesn't error
        _ = list(val_l)

    def test_large_batch_size_val_not_empty(self):
        # Even when batch_size > n_val_chunks, val_loader returns data
        # (drop_last=False for val)
        tokens = np.random.randint(0, 256, size=640, dtype=np.uint8)  # 10 chunks
        _, val_l = _to_loaders(tokens, seq_len=64, batch_size=100,
                                train_frac=0.8, random_state=0)
        batches = list(val_l)
        assert len(batches) > 0
