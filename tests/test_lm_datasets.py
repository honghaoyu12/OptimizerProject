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
