# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured data containers."""

from kumo_prior._sdm.tensor.var_len import VarLenTensor
from kumo_prior._sdm.tensor.string import StringTensor
from kumo_prior._sdm.tensor.nullable import NullableTensor
from kumo_prior._sdm.tensor.categorical import CategoricalTensor
from kumo_prior._sdm.tensor.columnar import ColumnarTensor
from kumo_prior._sdm.tensor.table import TableTensor

__all__ = [
    "VarLenTensor",
    "StringTensor",
    "NullableTensor",
    "CategoricalTensor",
    "ColumnarTensor",
    "TableTensor",
]
