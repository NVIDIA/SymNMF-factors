# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CUDA precision validation for the AdaptGrow dense path."""

import pytest
import torch

from adaptgrow import AdaptGrow, _sqnorm, _symnmf_error
from adaptgrow.corr_construction import corr_construct


pytestmark = pytest.mark.gpu


def _solve(matrix, allow_tf32):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    torch.manual_seed(42)
    solver = AdaptGrow(
        lr=2.0,
        entry_frac=1.0,
        max_iter=100,
        check_interval=10,
        verbose=False,
    )
    factors = solver.optimize(matrix, k=4)
    return _symnmf_error(matrix, factors, _sqnorm(matrix))


def test_tf32_preserves_small_matrix_reconstruction_error():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    matrix, _, _ = corr_construct(p=128, random_state=42)
    matrix = torch.as_tensor(matrix, dtype=torch.float32, device="cuda")
    fp32_error = _solve(matrix, allow_tf32=False)
    tf32_error = _solve(matrix, allow_tf32=True)

    relative_difference = abs(tf32_error - fp32_error) / max(fp32_error, 1e-12)
    assert relative_difference <= 0.05
