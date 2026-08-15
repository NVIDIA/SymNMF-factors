#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate GDS loading for a benchmark matrix.

This reads a contiguous row block into a CUDA tensor using
``adaptgrow.matrix_io``.
Use ``--method gds`` with ``KVIKIO_COMPAT_MODE=OFF`` to require real cuFile/GDS
rather than silently falling back to compatibility mode.
"""

import argparse
import importlib
import os
import pathlib
import time

import numpy as np
import torch

from adaptgrow.matrix_io import (
    get_last_load_info,
    load_matrix_rows_torch,
    read_matrix_meta,
)


def _module_status(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"missing ({type(exc).__name__}: {exc})"
    return f"ok ({getattr(module, '__version__', 'unknown version')})"


def main():
    parser = argparse.ArgumentParser(description="Check benchmark matrix GDS reads.")
    parser.add_argument("matrix_path")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--method", choices=("auto", "gds", "cpu"), default="gds")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    path = pathlib.Path(args.matrix_path)
    meta = read_matrix_meta(path)
    rows = min(args.rows, meta["p"])

    print(f"matrix={path}")
    print(f"p={meta['p']}, dtype={meta['dtype']}, rows={rows}")
    print(f"method={args.method}, device={args.device}")
    print(f"KVIKIO_COMPAT_MODE={os.environ.get('KVIKIO_COMPAT_MODE')}")
    print(f"cuda_available={torch.cuda.is_available()}, cuda_count={torch.cuda.device_count()}")
    for module_name in ("kvikio", "numba", "cupy"):
        print(f"{module_name}={_module_status(module_name)}")

    t0 = time.perf_counter()
    block = load_matrix_rows_torch(
        path,
        0,
        rows,
        device=args.device,
        method=args.method,
    )
    if block.device.type == "cuda":
        torch.cuda.synchronize(block.device)
    elapsed = time.perf_counter() - t0

    diag_n = min(rows, 16)
    diag = block[torch.arange(diag_n, device=block.device),
                 torch.arange(diag_n, device=block.device)]
    diag_ok = torch.allclose(diag, torch.ones_like(diag))

    cpu = np.memmap(path, dtype=meta["dtype"], mode="r",
                    shape=(meta["p"], meta["p"]))
    sample_cpu = np.array(cpu[: min(rows, 8), :8], copy=True)
    sample_gpu = block[: min(rows, 8), :8].detach().cpu().numpy()
    sample_ok = np.allclose(sample_cpu, sample_gpu)

    gb = block.numel() * block.element_size() / 1e9
    print(f"load_info={get_last_load_info()}")
    print(f"loaded shape={tuple(block.shape)}, dtype={block.dtype}, device={block.device}")
    print(f"bytes={block.numel() * block.element_size()}, GB={gb:.3f}, seconds={elapsed:.3f}")
    print(f"throughput={gb / elapsed:.3f} GB/s")
    print(f"diag_ok={bool(diag_ok)}, sample_ok={bool(sample_ok)}")


if __name__ == "__main__":
    main()
