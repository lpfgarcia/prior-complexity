"""Standalone copy of the NVIDIA Kumo Tabular synthetic prior, extracted from
https://github.com/NVIDIA/structured-data-models (commit 6d4a128):

* ``_sdm/``      — ``sdm.prior`` plus the parts of ``sdm`` it needs (stype, task, tensor)
* ``_builder.py`` — ``train/kumo/tabular/prior.py`` (the SCM configuration of Kumo)
* ``_loader.py``  — ``train/kumo/tabular/loader.py`` (row/column/train-size sampling,
  quality filters, standardisation)
* ``stages.py``   — the presets of the 18 training commands of ``train/kumo/tabular/README.md``

Licensed under Apache-2.0 (see ``LICENSE``); only import paths were changed.

Quick use::

    from kumo_prior import sample_datasets
    for ds in sample_datasets("clf", stage=1, n_datasets=10, seed=0):
        print(ds["X_train"].shape, ds["X_test"].shape)
"""

import warnings as _warnings

from ._loader import PriorDataset, SamplingGroup
from .stages import SIZES, STAGES, TASKS, get_stage_config
from .api import iter_batches, make_prior, sample_datasets, split_batch, stage_args

_warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*", category=UserWarning)

__version__ = "0.1.0"
__all__ = [
    "sample_datasets", "iter_batches", "make_prior", "split_batch", "stage_args",
    "PriorDataset", "SamplingGroup", "get_stage_config", "STAGES", "TASKS", "SIZES",
]
