# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal subset of ``sdm`` (https://github.com/NVIDIA/structured-data-models)
needed by the Kumo Tabular prior: the stype/task enums, the table tensors and
``sdm.prior``. The models, processing, relational and kernel packages are omitted.

Modified from the original ``sdm/__init__.py``: only the modules above are imported.
"""

from kumo_prior._sdm._constants import NaT
from kumo_prior._sdm.stype import Stype, StypeLike, infer_stypes
from kumo_prior._sdm.task import Task, TaskLike
from kumo_prior._sdm.tensor import (
    VarLenTensor,
    StringTensor,
    NullableTensor,
    CategoricalTensor,
    ColumnarTensor,
    TableTensor,
)
from kumo_prior._sdm import prior

__all__ = [
    "NaT",
    "Stype",
    "StypeLike",
    "Task",
    "TaskLike",
    "infer_stypes",
    "VarLenTensor",
    "StringTensor",
    "NullableTensor",
    "CategoricalTensor",
    "ColumnarTensor",
    "TableTensor",
    "prior",
]
