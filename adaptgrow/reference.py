# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run or verify a deterministic AdaptGrow reference configuration.

The runner is intentionally small: it exercises only the public AdaptGrow
implementation and synthetic matrix generators retained in this repository.
Wall-clock time is recorded for context but never used for result comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from . import AdaptGrow, _sqnorm, _symnmf_error
from .corr_construction import corr_construct
from .tpdm_construction import tpdm_construct


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "benchmark-small.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "reference" / "adaptgrow_summary.csv"


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _set_precision(allow_tf32: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _runtime_metadata(device: torch.device, allow_tf32: bool) -> dict[str, object]:
    nccl_version = None
    if device.type == "cuda":
        try:
            nccl_version = torch.cuda.nccl.version()
        except AttributeError:
            pass

    return {
        "git_revision": _git_revision(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": nccl_version,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "allow_tf32": allow_tf32,
        "container_image": os.environ.get("CONTAINER_IMAGE"),
    }


def load_config(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def run_reference(config: dict[str, object], device_name: str = "auto") -> tuple[dict[str, object], dict[str, object]]:
    matrix_config = config["matrix"]
    solver_config = config["solver"]
    seed = int(config["seed"])
    matrix_kind = matrix_config["kind"]
    size = int(matrix_config["size"])
    rank = int(solver_config["rank"])
    allow_tf32 = bool(solver_config["allow_tf32"])

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    constructors = {"corr": corr_construct, "tpdm": tpdm_construct}
    try:
        constructor = constructors[matrix_kind]
    except KeyError as error:
        raise ValueError(f"Unsupported matrix kind: {matrix_kind}") from error

    _set_precision(allow_tf32)
    matrix, _, _ = constructor(p=size, random_state=seed)
    matrix = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    matrix_sqnorm = _sqnorm(matrix)

    torch.manual_seed(seed)
    solver = AdaptGrow(
        lr=float(solver_config["learning_rate"]),
        max_iter=int(solver_config["max_iterations"]),
        entry_frac="auto",
        verbose=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    factors = solver.optimize(matrix, rank)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - start
    error = _symnmf_error(matrix, factors, matrix_sqnorm)

    row = {
        "name": config["name"],
        "matrix_kind": matrix_kind,
        "size": size,
        "seed": seed,
        "rank": rank,
        "learning_rate": solver.lr,
        "entry_frac": solver.resolved_["entry_frac"],
        "max_iterations": solver.max_iter,
        "converged": solver.converged_,
        "iterations": solver.n_iters_,
        "E_k": error,
        "wall_seconds": wall_seconds,
    }
    return row, _runtime_metadata(device, allow_tf32)


def write_result(row: dict[str, object], metadata: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def check_result(row: dict[str, object], expected_path: Path, relative_tolerance: float) -> None:
    with expected_path.open(newline="") as handle:
        expected = next(csv.DictReader(handle))
    expected_error = float(expected["E_k"])
    observed_error = float(row["E_k"])
    relative_difference = abs(observed_error - expected_error) / max(expected_error, 1e-12)
    if relative_difference > relative_tolerance:
        raise AssertionError(
            f"E_k changed by {relative_difference:.2%}; tolerance is {relative_tolerance:.2%}."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", type=Path, metavar="EXPECTED_CSV")
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args(argv)

    row, metadata = run_reference(load_config(args.config), args.device)
    if args.check is not None:
        check_result(row, args.check, args.rtol)
        print(f"Reference check passed against {args.check}.")
        return

    write_result(row, metadata, args.output)
    print(f"Wrote reference result to {args.output}.")


if __name__ == "__main__":
    main()
