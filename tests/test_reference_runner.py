# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the reproducible AdaptGrow reference runner."""

from adaptgrow.reference import run_reference


def test_reference_runner_records_solver_and_runtime_metadata():
    config = {
        "name": "test-reference",
        "seed": 42,
        "matrix": {"kind": "corr", "size": 16},
        "solver": {
            "name": "AdaptGrow",
            "rank": 4,
            "learning_rate": 1.0,
            "max_iterations": 3,
            "precision": "float32",
            "allow_tf32": False,
        },
        "runtime": {"device": "cpu-or-cuda"},
    }

    row, metadata = run_reference(config, device_name="cpu")

    assert row["name"] == "test-reference"
    assert row["matrix_kind"] == "corr"
    assert row["E_k"] >= 0
    assert metadata["device"] == "cpu"
    assert metadata["allow_tf32"] is False
