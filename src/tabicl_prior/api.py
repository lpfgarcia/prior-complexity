"""In-process Python API to sample TabICLv2 prior datasets for a given stage.

Examples
--------
>>> from tabicl_prior import sample_datasets
>>> for ds in sample_datasets("clf", stage=1, n_datasets=10, seed=0):
...     X_train, y_train, X_test, y_test = ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"]

>>> from tabicl_prior import iter_batches          # raw batches, like the trainer sees them
>>> for X, y, d, seq_lens, train_sizes in iter_batches("reg", stage=2, n_batches=5, n_jobs=4):
...     ...

>>> from tabicl_prior import make_prior             # the underlying PriorDataset
>>> prior = make_prior("clf", stage=3, batch_size=8, max_seq_len=20_000)
>>> X, y, d, seq_lens, train_sizes = prior.get_batch()

Note: with ``n_jobs > 0`` worker processes are used. On macOS/Windows (spawn start
method) call these functions under ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import math
import random
import warnings
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tabicl_prior._dataset import PriorDataset
from tabicl_prior.stages import STAGES, TASKS

warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*", category=UserWarning)

__all__ = ["stage_args", "make_prior", "iter_batches", "sample_datasets", "split_batch"]

_NON_OVERRIDABLE = {"task", "stage", "show_config", "train_steps"}


def stage_args(task: str, stage: int, **overrides):
    """Effective options (argparse.Namespace) of a stage preset, with overrides applied.

    Any option of the CLI can be overridden by name, e.g. ``max_seq_len=2048``,
    ``batch_size=8``, ``filter_unpredictable_datasets=False``.
    """
    from tabicl_prior.generate import parse_args  # local import: avoids a cycle

    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    args = parse_args(["--task", task, "--stage", str(stage)])
    for key, value in overrides.items():
        if key in _NON_OVERRIDABLE or not hasattr(args, key):
            raise TypeError(f"unknown or non-overridable option {key!r}")
        setattr(args, key, value)
    return args


def make_prior(task: str, stage: int, **overrides) -> PriorDataset:
    """Build the :class:`PriorDataset` of a stage. ``prior.get_batch()`` returns one batch.

    ``get_batch`` uses the global ``random`` / ``numpy`` / ``torch`` RNGs, so seed them
    yourself if you call it directly (or use :func:`iter_batches` with ``seed=``).
    """
    from tabicl_prior.generate import make_prior as _make_prior

    return _make_prior(stage_args(task, stage, **overrides))


class _PrivateRNG:
    """Own random/numpy/torch RNG state, swapped in only while generating, so that
    seeding the prior does not disturb the RNG state of the caller's program."""

    def __init__(self, seed: int):
        saved = self._get()
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        self.state = self._get()
        self._set(saved)

    @staticmethod
    def _get():
        return random.getstate(), np.random.get_state(), torch.get_rng_state()

    @staticmethod
    def _set(state):
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])

    def __enter__(self):
        self.saved = self._get()
        self._set(self.state)

    def __exit__(self, *exc):
        self.state = self._get()
        self._set(self.saved)


def _quiet_seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def iter_batches(
    task: str,
    stage: int,
    n_batches: Optional[int] = None,
    seed: Optional[int] = None,
    n_jobs: int = 0,
    **overrides,
) -> Iterator[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Yield raw batches ``(X, y, d, seq_lens, train_sizes)`` exactly as the trainer gets them.

    Parameters
    ----------
    task : {"clf", "reg"}
    stage : {1, 2, 3}
    n_batches : int, optional
        Number of batches; ``None`` = infinite stream.
    seed : int, optional
        Seed for reproducibility. ``None`` = fresh randomness every call.
        The caller's global RNG state is left untouched.
    n_jobs : int, default=0
        0 = generate in the current process; >0 = number of worker processes.
    **overrides
        Any stage option, e.g. ``batch_size=8`` (default 64), ``max_seq_len=2048``.

    Yields
    ------
    X : nested Tensor of B tensors (seq_len_i, max_features), zero-padded after d_i columns
        (a dense (B, T, max_features) tensor if ``seq_len_per_gp=False``)
    y : nested Tensor of B tensors (seq_len_i,) — int64 classes or float32 targets
    d, seq_lens, train_sizes : Tensor (B,) — #features, #rows, #train rows per dataset
    """
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    prior = make_prior(task, stage, **overrides)
    count = range(n_batches) if n_batches is not None else iter(int, 1)

    if n_jobs <= 0:
        rng = _PrivateRNG(seed)
        for _ in count:
            with rng:
                batch = prior.get_batch()
            yield batch
        return

    loader = DataLoader(
        prior,
        batch_size=None,
        num_workers=n_jobs,
        prefetch_factor=2,
        worker_init_fn=_quiet_seed_worker,
        generator=torch.Generator().manual_seed(seed),  # fixes the workers' base seed
    )
    it = iter(loader)
    try:
        for _ in count:
            yield next(it)
    finally:
        del it  # shuts the worker processes down


def split_batch(X, y, d, seq_lens, train_sizes, split: bool = True) -> Iterator[dict]:
    """Turn one raw batch into per-dataset numpy dicts with the padding removed."""
    nested = bool(getattr(X, "is_nested", False))
    for i in range(len(d)):
        T, di, ts = int(seq_lens[i]), int(d[i]), int(train_sizes[i])
        Xi = (X[i] if nested else X[i]).cpu()[:T, :di].numpy()
        yi = (y[i] if nested else y[i]).cpu()[:T].numpy()
        out = {"X": Xi, "y": yi, "train_size": ts}
        if split:
            out.update(X_train=Xi[:ts], y_train=yi[:ts], X_test=Xi[ts:], y_test=yi[ts:])
        yield out


def sample_datasets(
    task: str,
    stage: int,
    n_datasets: int,
    seed: Optional[int] = None,
    n_jobs: int = 0,
    split: bool = True,
    **overrides,
) -> Iterator[dict]:
    """Yield ``n_datasets`` datasets of a stage as numpy dicts.

    Each dict has ``X`` (n_samples, n_features) float32, ``y`` (n_samples,)
    (int64 classes for ``"clf"``, float32 for ``"reg"``), ``train_size`` (int) and,
    if ``split=True``, ``X_train, y_train, X_test, y_test``.

    Datasets are produced in batches of ``batch_size`` (default 64; datasets that share
    a group have the same number of rows and features). Use a smaller ``batch_size``
    override to get the first datasets sooner. See :func:`iter_batches` for the other
    parameters.
    """
    batch_size = overrides.get("batch_size") or stage_args(task, stage).batch_size
    n_batches = math.ceil(n_datasets / batch_size)
    produced = 0
    for batch in iter_batches(task, stage, n_batches=n_batches, seed=seed, n_jobs=n_jobs, **overrides):
        for ds in split_batch(*batch, split=split):
            yield ds
            produced += 1
            if produced >= n_datasets:
                return
