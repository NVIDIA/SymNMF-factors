# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone AdaptGrow SymNMF implementation.

Public API:
    AdaptGrow, AdaGrow, SymNMF_AdaGrad, SymNMF_BlockSVRG, spectral_probe

The three solvers live in their own modules:
    * ``adaptgrow.adagrad``    -- :class:`SymNMF_AdaGrad` (full-batch, phi = 1)
    * ``adaptgrow.block_svrg`` -- :class:`SymNMF_BlockSVRG` (stochastic, phi < 1)
    * ``adaptgrow.adaptgrow``  -- :class:`AdaptGrow` (orchestrator)

The implementation accepts a dense non-negative symmetric ``torch.Tensor`` or
a duck-typed distributed matrix supplied by ``adaptgrow.distributed``.
"""

from ._core import (
    DENSE_FRAC,
    _LR_AUTO_MSG,
    _block_grad,
    _block_sample,
    _final_report,
    _grad,
    _init_solve,
    _projected_grad_norm,
    _run_block,
    _run_dense,
    _sqnorm,
    _symnmf_error,
    spectral_probe,
)
from .adagrad import SymNMF_AdaGrad
from .adaptgrow import AdaGrow, AdaptGrow
from .block_svrg import SymNMF_BlockSVRG

__all__ = [
    "AdaptGrow",
    "AdaGrow",
    "SymNMF_AdaGrad",
    "SymNMF_BlockSVRG",
    "spectral_probe",
]
