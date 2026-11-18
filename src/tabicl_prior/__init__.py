"""Standalone copy of the TabICLv2 synthetic prior (graph_scm), extracted from
https://github.com/soda-inria/tabicl (``src/tabicl/prior``, commit a626a1b).

Quick use::

    from tabicl_prior import sample_datasets
    for ds in sample_datasets("clf", stage=1, n_datasets=10, seed=0):
        print(ds["X_train"].shape, ds["X_test"].shape)
"""

from ._dataset import PriorDataset
from .graph_lib._config import PriorConfig
from .stages import STAGES, TASKS, get_stage_config
from .io import iter_datasets, LoadPriorDataset
from .api import make_prior, iter_batches, sample_datasets, split_batch, stage_args

__version__ = "0.1.0"
__all__ = [
    "sample_datasets", "iter_batches", "make_prior", "split_batch", "stage_args",
    "PriorDataset", "PriorConfig", "get_stage_config", "STAGES", "TASKS", "iter_datasets", "LoadPriorDataset"]
