"""Presets of the Kumo Tabular pre-training prior for each curriculum stage.

The values are copied from the 18 training commands in ``train/kumo/tabular/README.md``
of https://github.com/NVIDIA/structured-data-models (commit 6d4a128, Nov 2026),
keeping only the arguments that affect the prior. Every other option keeps the
default of ``train/kumo/tabular/train.py``.

Curriculum (paper arXiv:2611.23001):

=======  ===========================  ==============  =========================================
Stage    Rows per dataset             Train fraction  Notes
=======  ===========================  ==============  =========================================
1        1,024 (fixed)                30%–90%         relabel/coarsen/high-cardinality/near-empty/
                                                      latent missingness/softmax-argmax off,
                                                      trivial datasets always rejected
2        400–10,240 (log-uniform)     79%–81%         every augmentation at its default probability
3        400–60,000 (log-uniform)     79%–81%         ``replay_small`` + size-specific column caps
=======  ===========================  ==============  =========================================

Classification and regression use the same SCM; regression keeps the target continuous
(``y_heavy_tail_prob`` is 0 in stages 1–2 and 0.2 in stage 3). The model size only
changes the batch layout (``batch_size``, ``group_size``), the number of steps and,
in stage 3, the column cap per row count (``max_columns_ladder``). ``"large"`` is the
default because it is the default size of the released model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

TASKS = ("clf", "reg")
STAGES = (1, 2, 3)
SIZES = ("small", "medium", "large")

# Default column caps of train/kumo/tabular/loader.py (used in stages 1 and 2, where
# every dataset has at most 10,240 rows, so the cap is never reached).
DEFAULT_MAX_COLUMNS_LADDER = (
    (10_240, None),
    (20_000, 80),
    (30_000, 60),
    (40_000, 40),
    (50_000, 30),
    (60_000, 20),
    (65_000, 15),
    (None, 10),
)

# Defaults of train/kumo/tabular/train.py for the options that affect the prior.
_DEFAULTS: Dict[str, Any] = dict(
    min_rows=1024,
    max_rows=1024,
    min_train_size=0.3,
    max_train_size=0.9,
    batch_size=64,
    group_size=8,
    replay_small=False,
    max_columns_ladder=DEFAULT_MAX_COLUMNS_LADDER,
    keep_trivial_prob=0.5,
    correlated_block_prob=0.3,
    relabel_prob=0.1,
    coarsen_prob=0.1,
    high_cardinality_prob=0.05,
    y_heavy_tail_prob=0.2,
    structured_missingness_prob=0.2,
    mcar_missingness_prob=0.01,
    near_empty_missingness_prob=0.05,
    latent_driver_missingness_prob=0.03,
    softmax_argmax_prob=0.15,
    min_regression_train_std=0.1,
    seed=0,
)

# Stage 1 switches these augmentations off (and keep_trivial_prob=0).
_STAGE1_OFF = dict(
    relabel_prob=0.0,
    coarsen_prob=0.0,
    high_cardinality_prob=0.0,
    near_empty_missingness_prob=0.0,
    latent_driver_missingness_prob=0.0,
    softmax_argmax_prob=0.0,
    keep_trivial_prob=0.0,
)

_PER_STAGE: Dict[int, Dict[str, Any]] = {
    1: dict(min_rows=1024, max_rows=1024, min_train_size=0.3, max_train_size=0.9, **_STAGE1_OFF),
    2: dict(min_rows=400, max_rows=10_240, min_train_size=0.79, max_train_size=0.81),
    3: dict(min_rows=400, max_rows=60_000, min_train_size=0.79, max_train_size=0.81, replay_small=True),
}

_LADDER_MEDIUM = (
    (16_000, 100), (20_000, 100), (25_000, 100), (32_000, 100), (36_000, 88), (40_000, 80),
    (45_000, 72), (50_000, 64), (55_000, 58), (60_000, 53), (None, 48),
)
_LADDER_LARGE = (
    (16_000, 100), (20_000, 100), (25_000, 100), (32_000, 78), (36_000, 69), (40_000, 61),
    (45_000, 55), (50_000, 48), (55_000, 44), (60_000, 40), (None, 40),
)

# (size, stage) -> batch layout, steps and stage-3 column caps.
_PER_SIZE: Dict[tuple, Dict[str, Any]] = {
    ("small", 1): dict(batch_size=64, group_size=8, train_steps=500_000),
    ("small", 2): dict(batch_size=128, group_size=4, train_steps=40_000),
    ("small", 3): dict(batch_size=128, group_size=2, train_steps=12_000, max_columns_ladder=((None, 100),)),
    ("medium", 1): dict(batch_size=128, group_size=2, train_steps=500_000),
    ("medium", 2): dict(batch_size=128, group_size=1, train_steps=40_000),
    ("medium", 3): dict(batch_size=128, group_size=1, train_steps=12_000, max_columns_ladder=_LADDER_MEDIUM),
    ("large", 1): dict(batch_size=128, group_size=2, train_steps=1_000_000),
    ("large", 2): dict(batch_size=128, group_size=1, train_steps=60_000),
    ("large", 3): dict(batch_size=128, group_size=1, train_steps=18_000, max_columns_ladder=_LADDER_LARGE),
}

_PER_TASK: Dict[str, Dict[str, Any]] = {
    "clf": dict(task_type="classification", seed=42),
    "reg": dict(task_type="regression", seed=42),
}

_REG_PER_STAGE: Dict[int, Dict[str, Any]] = {
    1: dict(y_heavy_tail_prob=0.0),
    2: dict(y_heavy_tail_prob=0.0),
    3: dict(y_heavy_tail_prob=0.2),
}


def get_stage_config(task: str, stage: int, size: str = "large") -> Dict[str, Any]:
    """Return the prior configuration used to pre-train Kumo Tabular.

    Parameters
    ----------
    task : {"clf", "reg"}
    stage : {1, 2, 3}
    size : {"small", "medium", "large"}, default="large"

    Returns
    -------
    dict
        Flat dict of generator options. ``train_steps`` is informational: the number
        of optimizer steps (= batches of ``batch_size`` datasets) of that stage.
        ``batch_size`` is the global batch size (the per-step number of datasets).
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, got {size!r}")
    cfg = deepcopy(_DEFAULTS)
    cfg.update(deepcopy(_PER_STAGE[stage]))
    cfg.update(deepcopy(_PER_SIZE[(size, stage)]))
    cfg.update(deepcopy(_PER_TASK[task]))
    if task == "reg":
        cfg.update(_REG_PER_STAGE[stage])
    cfg.update(task=task, stage=stage, size=size)
    return cfg
