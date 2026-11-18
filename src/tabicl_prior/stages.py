"""Presets of the TabICLv2 pre-training prior for each curriculum stage.

The values below are copied from the official training scripts
``scripts/train_v2_{clf,reg}_stage{1,2,3}.sh`` of https://github.com/soda-inria/tabicl
(commit a626a1b, Nov 2026), keeping only the arguments that affect the prior.
Every other prior option keeps the default of ``tabicl.train`` (which is the same
as the defaults of :class:`PriorConfig` and of the ``generate`` CLI).

Curriculum (paper arXiv:2602.11139, §4.1 / App. B.1), batch size 64 throughout:

=======  ==========  =========================  =============  ==================
Stage    Steps       Samples per dataset        Train fraction Datasets in total
=======  ==========  =========================  =============  ==================
1        500,000     1,024 (fixed)              30%–90%        32,000,000
2         40,000     400–10,240 (log-uniform)   79%–81%         2,560,000
3         10,000     400–60,000 (log-uniform)   79%–81%           640,000
=======  ==========  =========================  =============  ==================

Classifier and regressor use the same prior settings; the only differences are
``regression=True`` (target kept continuous, outlier-removed and standardised) and
the seed (42 for the classifier runs, 43 for the regressor runs).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

TASKS = ("clf", "reg")
STAGES = (1, 2, 3)

# Options shared by every stage and both tasks.
_COMMON: Dict[str, Any] = dict(
    prior_type="graph_scm",
    batch_size=64,
    min_features=1,
    max_features=100,
    max_classes=10,
    seq_len_per_gp=True,
    # graph_scm (PriorConfig) options
    graph_noise=False,
    filter_unpredictable_graphs=True,
    filter_unpredictable_datasets=True,
    allow_act_warping=False,
    min_n_nodes=2,
    max_n_nodes=32,
    cauchy_dag_offset=0.0,
)

_PER_STAGE: Dict[int, Dict[str, Any]] = {
    1: dict(
        batch_size_per_gp=4,
        min_seq_len=None,  # None => every dataset has exactly max_seq_len samples
        max_seq_len=1024,
        log_seq_len=False,
        min_train_size=0.3,
        max_train_size=0.9,
        train_steps=500_000,
    ),
    2: dict(
        batch_size_per_gp=1,
        min_seq_len=400,
        max_seq_len=10_240,
        log_seq_len=True,
        min_train_size=0.79,
        max_train_size=0.81,
        train_steps=40_000,
    ),
    3: dict(
        batch_size_per_gp=1,
        min_seq_len=400,
        max_seq_len=60_000,
        log_seq_len=True,
        min_train_size=0.79,
        max_train_size=0.81,
        train_steps=10_000,
    ),
}

_PER_TASK: Dict[str, Dict[str, Any]] = {
    "clf": dict(regression=False, seed=42),
    "reg": dict(regression=True, seed=43),
}


def get_stage_config(task: str, stage: int) -> Dict[str, Any]:
    """Return the prior configuration used to pre-train TabICLv2.

    Parameters
    ----------
    task : {"clf", "reg"}
        Classifier or regressor checkpoint.
    stage : {1, 2, 3}
        Curriculum stage.

    Returns
    -------
    dict
        Flat dict of generator options. ``train_steps`` is informational: the number
        of optimizer steps (= batches of ``batch_size`` datasets) of that stage.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    cfg = deepcopy(_COMMON)
    cfg.update(deepcopy(_PER_STAGE[stage]))
    cfg.update(deepcopy(_PER_TASK[task]))
    cfg["task"] = task
    cfg["stage"] = stage
    return cfg
