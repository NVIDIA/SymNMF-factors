# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast CPU checks for the documented public workflow."""

from pathlib import Path

import torch

from adaptgrow import AdaptGrow
from adaptgrow.benchmarks import generate_all


def test_adaptgrow_factorizes_small_symmetric_matrix():
    torch.manual_seed(42)
    reference = torch.rand(12, 3)
    matrix = reference @ reference.T

    solver = AdaptGrow(
        lr=1.0,
        entry_frac=1.0,
        max_iter=5,
        check_interval=1,
        verbose=False,
    )
    factors = solver.optimize(matrix, k=3)

    assert factors.shape == (12, 3)
    assert torch.all(factors >= 0)
    assert solver.resolved_["entry_frac"] == 1.0


def test_small_matrix_generation_writes_companion_files(tmp_path: Path):
    generate_all(
        sizes=[10],
        seeds=[42],
        output_dir=tmp_path,
        matrix_types=["corr"],
    )

    assert (tmp_path / "corr_p10_s42.dat").exists()
    assert (tmp_path / "corr_p10_s42.dat.meta.json").exists()
    assert (tmp_path / "corr_p10_s42.dat.labels.npy").exists()
