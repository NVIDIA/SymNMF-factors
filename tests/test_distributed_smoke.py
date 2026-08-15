# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in two-GPU smoke test for distributed AdaptGrow."""

import subprocess
import sys

import pytest
import torch

from adaptgrow.corr_construction import corr_construct


pytestmark = [pytest.mark.gpu, pytest.mark.distributed]


def test_two_gpu_runner_factorizes_generated_matrix(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")

    matrix_path = tmp_path / "corr_p32_s42.dat"
    corr_construct(p=32, random_state=42, output_path=str(matrix_path))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "scripts/run_distributed.py",
        "--matrix-dir",
        str(tmp_path),
        "--kind",
        "corr",
        "--k",
        "4",
        "--n-iter",
        "5",
        "--io-method",
        "cpu",
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)

    assert result.returncode == 0, result.stdout + result.stderr
