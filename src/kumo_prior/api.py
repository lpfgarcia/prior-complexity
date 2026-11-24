"""In-process Python API to sample Kumo Tabular prior datasets for a given stage.

Same interface as :mod:`tabicl_prior.api`, plus a ``size`` argument (the stage presets
of Kumo differ slightly per model size, see :mod:`kumo_prior.stages`).

Examples
--------
>>> from kumo_prior import sample_datasets
>>> for ds in sample_datasets("clf", stage=1, n_datasets=10, seed=0):
...     X_train, y_train, X_test, y_test = ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"]

>>> from kumo_prior import iter_batches          # raw batches, like the trainer sees them
>>> for groups in iter_batches("reg", stage=2, n_batches=5, n_jobs=4):
...     for g in groups:                         # SamplingGroup(x, y, train_size, categorical_mask, ...)
...         ...

>>> from kumo_prior import make_prior             # the underlying PriorDataset
>>> prior = make_prior("clf", stage=3, size="small", batch_size=8)
>>> groups = next(iter(prior))

Note: with ``n_jobs > 0`` worker processes are used. On macOS/Windows (spawn start
method) call these functions under ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from kumo_prior._builder import make_prior as _make_scm_prior
from kumo_prior._loader import PriorDataset, SamplingGroup
from kumo_prior._sdm import Task
from kumo_prior.stages import get_stage_config

__all__ = ["stage_args", "make_prior", "iter_batches", "sample_datasets", "split_batch"]

_NON_OVERRIDABLE = {"task", "stage", "size", "task_type", "train_steps"}

# Arguments of make_prior() in train/kumo/tabular/prior.py.
_SCM_KEYS = (
    "correlated_block_prob",
    "relabel_prob",
    "coarsen_prob",
    "high_cardinality_prob",
    "y_heavy_tail_prob",
    "structured_missingness_prob",
    "mcar_missingness_prob",
    "near_empty_missingness_prob",
    "latent_driver_missingness_prob",
    "softmax_argmax_prob",
)


def stage_args(task: str, stage: int, size: str = "large", **overrides) -> dict:
    """Effective options of a stage preset, with overrides applied.

    Any option of :func:`kumo_prior.stages.get_stage_config` can be overridden by name,
    e.g. ``max_rows=2048``, ``batch_size=8``, ``relabel_prob=0``.
    """
    cfg = get_stage_config(task, stage, size)
    for key, value in overrides.items():
        if key in _NON_OVERRIDABLE or key not in cfg:
            raise TypeError(f"unknown or non-overridable option {key!r}")
        cfg[key] = value
    if cfg["batch_size"] % cfg["group_size"]:
        raise ValueError(f"batch_size ({cfg['batch_size']}) must be divisible by group_size ({cfg['group_size']})")
    return cfg


def _prior_dataset(cfg: dict, seed: int) -> PriorDataset:
    """Build the PriorDataset exactly as train/kumo/tabular/train.py does (one rank)."""
    scm = _make_scm_prior(Task(cfg["task_type"]), **{k: cfg[k] for k in _SCM_KEYS})
    return PriorDataset(
        prior=scm,
        batch_size=cfg["batch_size"],
        group_size=cfg["group_size"],
        min_rows=cfg["min_rows"],
        max_rows=cfg["max_rows"],
        min_train_size=cfg["min_train_size"],
        max_train_size=cfg["max_train_size"],
        replay_small=cfg["replay_small"],
        seed=seed,
        rank=0,
        world_size=1,
        keep_trivial_prob=cfg["keep_trivial_prob"],
        min_regression_train_std=cfg["min_regression_train_std"],
        robust_regression_probe=cfg["y_heavy_tail_prob"] > 0,
        max_columns_ladder=tuple(tuple(x) for x in cfg["max_columns_ladder"]),
    )


def make_prior(task: str, stage: int, size: str = "large", seed: Optional[int] = None, **overrides) -> PriorDataset:
    """Build the :class:`PriorDataset` of a stage. Iterating it yields one batch per step.

    Iterating re-seeds the global ``random`` / ``numpy`` / ``torch`` RNGs (as in the
    original trainer); use :func:`iter_batches` to keep the caller's RNG state intact.
    """
    cfg = stage_args(task, stage, size, **overrides)
    return _prior_dataset(cfg, cfg["seed"] if seed is None else seed)


class _PrivateRNG:
    """Own random/numpy/torch RNG state, swapped in only while generating, so that
    sampling from the prior does not disturb the RNG state of the caller's program."""

    def __init__(self):
        self.state = self._get()

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


def iter_batches(
    task: str,
    stage: int,
    n_batches: Optional[int] = None,
    seed: Optional[int] = None,
    n_jobs: int = 0,
    size: str = "large",
    **overrides,
) -> Iterator[List[SamplingGroup]]:
    """Yield raw batches exactly as the trainer gets them.

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
        0 = generate in the current process; >0 = number of worker processes
        (each worker draws its own stream, so results depend on ``n_jobs``).
    size : {"small", "medium", "large"}, default="large"
    **overrides
        Any stage option, e.g. ``batch_size=8``, ``max_rows=2048``.

    Yields
    ------
    list of SamplingGroup
        ``batch_size // group_size`` groups. Each group has ``x`` (group_size, rows,
        padded_columns) float32 standardised on the context rows (NaN = missing),
        ``y`` (group_size, rows) int64 classes or standardised float32 targets,
        ``train_size`` (int), ``categorical_mask`` (group_size, padded_columns) bool,
        ``num_columns`` (group_size,) and ``max_column_width``.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    prior = make_prior(task, stage, size, seed=seed, **overrides)
    count = range(n_batches) if n_batches is not None else iter(int, 1)

    if n_jobs <= 0:
        rng = _PrivateRNG()
        with rng:
            it = iter(prior)
        for _ in count:
            with rng:
                batch = next(it)
            yield batch
        return

    loader = DataLoader(prior, batch_size=None, num_workers=n_jobs, prefetch_factor=2)
    it = iter(loader)
    try:
        for _ in count:
            yield next(it)
    finally:
        del it  # shuts the worker processes down


def split_batch(groups: List[SamplingGroup], split: bool = True) -> Iterator[dict]:
    """Turn one raw batch into per-dataset numpy dicts with the column padding removed.

    Each dict has ``X`` (n_samples, n_features) float32 (may contain NaN: Kumo's prior
    injects missing values), ``y`` (n_samples,), ``train_size`` (int), ``categorical``
    (n_features,) bool mask and, if ``split=True``, ``X_train, y_train, X_test, y_test``.
    """
    for g in groups:
        ts = int(g.train_size)
        for i in range(g.x.size(0)):
            d = int(g.num_columns[i])
            Xi = g.x[i, :, :d].cpu().float().numpy()
            yi = g.y[i].cpu().numpy()
            out = {"X": Xi, "y": yi, "train_size": ts, "categorical": g.categorical_mask[i, :d].cpu().numpy()}
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
    size: str = "large",
    **overrides,
) -> Iterator[dict]:
    """Yield ``n_datasets`` datasets of a stage as numpy dicts (see :func:`split_batch`).

    Datasets are produced in batches of ``batch_size`` (64 or 128 depending on the
    stage and size); datasets of the same batch share the number of rows and those of
    the same group share the train size. Use a smaller ``batch_size`` override (a
    multiple of ``group_size``) to get the first datasets sooner. See
    :func:`iter_batches` for the other parameters.
    """
    batch_size = stage_args(task, stage, size, **overrides)["batch_size"]
    n_batches = math.ceil(n_datasets / batch_size)
    produced = 0
    for batch in iter_batches(task, stage, n_batches=n_batches, seed=seed, n_jobs=n_jobs, size=size, **overrides):
        for ds in split_batch(batch, split=split):
            yield ds
            produced += 1
            if produced >= n_datasets:
                return
