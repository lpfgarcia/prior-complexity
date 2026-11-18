"""Read prior data written by :mod:`tabicl_prior.generate` (or ``python -m tabicl.prior``).

Two ways to consume a directory:

* :func:`iter_datasets` — one dataset at a time as numpy arrays, padding removed
  (handy for analysis, export, or training your own model).
* :class:`LoadPriorDataset` — the original batched loader used by ``tabicl.train``
  (re-exported here, with the nested-tensor loading fix).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from tabicl_prior._genload import LoadPriorDataset, load_batch_file, sparse2dense

__all__ = ["iter_datasets", "iter_batch_files", "read_metadata", "load_batch_file", "LoadPriorDataset"]


def read_metadata(data_dir) -> dict:
    f = Path(data_dir) / "metadata.json"
    return json.loads(f.read_text()) if f.exists() else {}


def iter_batch_files(data_dir) -> Iterator[Path]:
    yield from sorted(Path(data_dir).glob("batch_*.pt"))


def _unpack(batch: dict):
    """Yield (X, y, d, seq_len, train_size) per dataset as torch tensors."""
    X, y, d = batch["X"], batch["y"], batch["d"]
    seq_lens, train_sizes, B = batch["seq_lens"], batch["train_sizes"], batch["batch_size"]
    if not X.is_nested:  # fixed seq_len: X is stored flat (sparse), y dense (B, T)
        T = int(seq_lens[0])
        X = sparse2dense(X, d.repeat_interleave(T), dtype=torch.float32).view(B, T, -1)
    for i in range(B):
        yield X[i], y[i], int(d[i]), int(seq_lens[i]), int(train_sizes[i])


def iter_datasets(data_dir, max_datasets: Optional[int] = None, split: bool = False) -> Iterator[dict]:
    """Iterate over every dataset stored in ``data_dir``.

    Parameters
    ----------
    data_dir : path
        Directory with ``batch_*.pt`` files.
    max_datasets : int, optional
        Stop after this many datasets.
    split : bool, default=False
        If True, also return ``X_train, y_train, X_test, y_test``.

    Yields
    ------
    dict with keys
        ``X`` (n_samples, n_features) float32, ``y`` (n_samples,) int64 for
        classification / float32 for regression, ``train_size`` (int: the first
        ``train_size`` rows are the in-context training set, the rest are test rows),
        ``batch_file``, ``index_in_batch``; plus the split arrays if ``split=True``.
    """
    n = 0
    for f in iter_batch_files(data_dir):
        batch = load_batch_file(f)
        for j, (X, y, d, T, ts) in enumerate(_unpack(batch)):
            Xn = X[:T, :d].numpy()
            yn = y[:T].numpy()
            out = {"X": Xn, "y": yn, "train_size": ts, "batch_file": f.name, "index_in_batch": j}
            if split:
                out.update(X_train=Xn[:ts], y_train=yn[:ts], X_test=Xn[ts:], y_test=yn[ts:])
            yield out
            n += 1
            if max_datasets is not None and n >= max_datasets:
                return
