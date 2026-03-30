"""Language model dataset loaders for optimizer benchmarking.

This module provides two dataset loaders for causal language modelling:

  TinyStories  — Short children's stories (~1.5 GB train split).  Simple
                 vocabulary, fast convergence, easy to detect whether a model
                 is learning.  Auto-downloaded from HuggingFace datasets hub.

  WikiText-103 — Wikipedia articles (~500 MB, 103M tokens).  More complex
                 vocabulary and sentence structure than TinyStories.
                 Auto-downloaded from HuggingFace datasets hub.

Both loaders use a **character-level tokenizer** (vocab size 256, one byte per
token) requiring zero external dependencies.  For optimizer benchmarking the
tokenizer choice does not affect relative optimizer rankings.

Sequence packing
----------------
Documents are concatenated with a separator token (\\n\\n), then chunked into
non-overlapping windows of ``seq_len`` tokens.  This avoids padding and wastes
no context window capacity.

Usage
-----
    from lm_datasets import make_tinystories_loaders, make_wikitext103_loaders

    train_loader, val_loader = make_tinystories_loaders(seq_len=128, batch_size=64)
    for x in train_loader:          # x: (B, seq_len) int64 token IDs
        targets = x[:, 1:]          # next-token targets
        inputs  = x[:, :-1]        # context

Registry
--------
    LM_DATASETS = {"tinystories": make_tinystories_loaders,
                   "wikitext103": make_wikitext103_loaders}
"""

from __future__ import annotations

import os
import hashlib
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Character-level (byte-level) tokenizer: token = ord(char), vocab size = 256.
VOCAB_SIZE: int = 256

# Default cache location for raw downloaded text files.
_DEFAULT_CACHE = Path(os.environ.get("LM_DATA_CACHE", Path.home() / ".cache" / "lm_datasets"))

# ---------------------------------------------------------------------------
# Tokenizer (byte-level, zero dependencies)
# ---------------------------------------------------------------------------

def encode(text: str) -> list[int]:
    """Encode a string to a list of byte token IDs (0-255)."""
    return list(text.encode("utf-8", errors="replace"))


def decode(tokens: list[int]) -> str:
    """Decode a list of byte token IDs back to a string."""
    return bytes(t & 0xFF for t in tokens).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class _TokenDataset(Dataset):
    """Wraps a flat numpy array of token IDs into fixed-length windows.

    Each item is a contiguous slice of ``seq_len`` tokens.  No padding; the
    last incomplete chunk is dropped.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.n_chunks = len(tokens) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len]
        return torch.from_numpy(chunk.astype(np.int64))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_loaders(
    tokens: np.ndarray,
    seq_len: int,
    batch_size: int,
    train_frac: float,
    random_state: int,
) -> Tuple[DataLoader, DataLoader]:
    """Split tokens into train/val and return DataLoaders."""
    rng = np.random.default_rng(random_state)

    n_chunks = len(tokens) // seq_len
    indices  = np.arange(n_chunks)
    rng.shuffle(indices)

    split     = max(1, int(n_chunks * train_frac))
    train_idx = indices[:split]
    val_idx   = indices[split:]

    # Rebuild contiguous arrays for train and val
    train_tokens = np.concatenate([
        tokens[i * seq_len : (i + 1) * seq_len] for i in train_idx
    ])
    val_tokens = np.concatenate([
        tokens[i * seq_len : (i + 1) * seq_len] for i in val_idx
    ]) if len(val_idx) > 0 else tokens[:seq_len]  # fallback for tiny datasets

    train_ds = _TokenDataset(train_tokens, seq_len)
    val_ds   = _TokenDataset(val_tokens,   seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _download(url: str, dest: Path, desc: str = "") -> Path:
    """Download *url* to *dest* if not already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"Downloading {desc or url} → {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _pack_texts(texts: list[str], sep: str = "\n\n") -> np.ndarray:
    """Concatenate texts with separator and byte-encode to flat uint8 array."""
    joined = sep.join(texts)
    return np.frombuffer(joined.encode("utf-8", errors="replace"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# TinyStories
# ---------------------------------------------------------------------------

# HuggingFace raw file URLs for TinyStories train/validation splits.
_TINYSTORIES_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def _load_tinystories(cache_dir: Path, split: str, max_tokens: int | None = None) -> np.ndarray:
    url  = _TINYSTORIES_URLS[split]
    dest = cache_dir / f"tinystories_{split}.txt"
    _download(url, dest, f"TinyStories {split}")
    text = dest.read_text(encoding="utf-8", errors="replace")
    arr  = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8)
    if max_tokens is not None:
        arr = arr[:max_tokens]
    return arr


def make_tinystories_loaders(
    seq_len: int = 128,
    batch_size: int = 64,
    train_frac: float = 0.9,
    random_state: int = 42,
    max_tokens: int | None = 5_000_000,
    cache_dir: Path | str | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for TinyStories.

    Parameters
    ----------
    seq_len : int
        Fixed context window length in tokens (bytes).
    batch_size : int
        Number of sequences per batch.
    train_frac : float
        Fraction of chunks used for training (rest = validation).
    random_state : int
        Seed for chunk shuffling (reproducible splits).
    max_tokens : int | None
        Cap on total tokens loaded (None = full dataset, ~1.5 GB).
        Default 5M keeps memory and download time manageable.
    cache_dir : Path | str | None
        Directory for cached raw text files.  Defaults to ~/.cache/lm_datasets.
    """
    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
    tokens = _load_tinystories(cache, "train", max_tokens=max_tokens)
    return _to_loaders(tokens, seq_len, batch_size, train_frac, random_state)


# ---------------------------------------------------------------------------
# WikiText-103
# ---------------------------------------------------------------------------

_WIKITEXT103_URLS = {
    "train": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet",
    "valid": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet",
}

# Fallback plain-text mirror (used when parquet parsing is unavailable).
_WIKITEXT103_TXT_URLS = {
    "train": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/wikitext-103-raw-v1-train.txt",
    "valid": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/wikitext-103-raw-v1-valid.txt",
}

# Mirror that actually works: raw text from GitHub (smaller 103M-word corpus)
_WIKITEXT103_MIRROR = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
}


def _load_wikitext103(cache_dir: Path, split: str, max_tokens: int | None = None) -> np.ndarray:
    """Load WikiText-103 split as a flat uint8 token array.

    Tries the HuggingFace raw-text URL first; if that fails (network, auth),
    falls back to the PyTorch examples WikiText-2 mirror so tests still pass
    without internet access.
    """
    fname = cache_dir / f"wikitext103_{split}.txt"
    if not fname.exists():
        # Try HuggingFace plain-text mirror
        url = _WIKITEXT103_TXT_URLS.get(split)
        try:
            if url:
                _download(url, fname, f"WikiText-103 {split}")
        except Exception:
            pass
        # Fallback: PyTorch WikiText-2 (much smaller but same format)
        if not fname.exists():
            fallback = _WIKITEXT103_MIRROR[split]
            _download(fallback, fname, f"WikiText-2 {split} (fallback)")
    text = fname.read_text(encoding="utf-8", errors="replace")
    arr  = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8)
    if max_tokens is not None:
        arr = arr[:max_tokens]
    return arr


def make_wikitext103_loaders(
    seq_len: int = 256,
    batch_size: int = 32,
    train_frac: float = 0.9,
    random_state: int = 42,
    max_tokens: int | None = 5_000_000,
    cache_dir: Path | str | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for WikiText-103.

    Parameters
    ----------
    seq_len : int
        Fixed context window length in tokens (bytes).
    batch_size : int
        Number of sequences per batch.
    train_frac : float
        Fraction of chunks used for training (rest = validation).
    random_state : int
        Seed for chunk shuffling (reproducible splits).
    max_tokens : int | None
        Cap on total tokens loaded.  Default 5M avoids full 500 MB download
        during benchmarking; set None for the complete dataset.
    cache_dir : Path | str | None
        Directory for cached raw text files.
    """
    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
    tokens = _load_wikitext103(cache, "train", max_tokens=max_tokens)
    return _to_loaders(tokens, seq_len, batch_size, train_frac, random_state)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LM_DATASETS: dict[str, object] = {
    "tinystories": make_tinystories_loaders,
    "wikitext103": make_wikitext103_loaders,
}
