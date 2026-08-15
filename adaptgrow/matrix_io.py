# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix I/O helpers for dense benchmark memmaps.

The benchmark generators write raw row-major float32 files with companion
``.meta.json`` metadata.  The default loader path stages through CPU memory.
For large matrices on GPU nodes, this module can instead use KvikIO/cuFile
to read directly into GPU memory via GPUDirect Storage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch


LoadMethod = Literal["auto", "gds", "cpu"]
_LAST_LOAD_INFO = {}


def get_last_load_info():
    """Return metadata for the most recent matrix load in this process."""
    return dict(_LAST_LOAD_INFO)


def _record_load_info(**info):
    _LAST_LOAD_INFO.clear()
    _LAST_LOAD_INFO.update(info)


def _format_exception(exc):
    return f"{type(exc).__name__}: {exc}"


def _require_kvikio():
    import kvikio
    return kvikio


def read_matrix_meta(matrix_path):
    """Read companion metadata for a raw benchmark memmap."""
    matrix_path = Path(matrix_path)
    with open(str(matrix_path) + ".meta.json") as f:
        meta = json.load(f)
    p = int(meta["p"])
    dtype = np.dtype(meta.get("dtype", "float32"))
    return {"p": p, "dtype": dtype}


def expected_matrix_nbytes(matrix_path):
    """Expected raw file size from companion metadata."""
    meta = read_matrix_meta(matrix_path)
    return meta["p"] * meta["p"] * meta["dtype"].itemsize


def _validate_file_size(matrix_path):
    matrix_path = Path(matrix_path)
    expected = expected_matrix_nbytes(matrix_path)
    actual = matrix_path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{matrix_path} has {actual} bytes, expected {expected}. "
            "The file may be incomplete or metadata may be stale."
        )


def _torch_from_cupy(cupy_array):
    """Create a torch tensor sharing a CuPy allocation."""
    try:
        return torch.utils.dlpack.from_dlpack(cupy_array)
    except TypeError:
        return torch.utils.dlpack.from_dlpack(cupy_array.toDlpack())


def _read_with_kvikio(path, device_array, file_offset=0, nbytes=None):
    """Read into a contiguous CUDA array using KvikIO/cuFile."""
    kvikio = _require_kvikio()

    if nbytes is None:
        nbytes = device_array.nbytes

    flat = device_array.reshape(-1)
    with kvikio.CuFile(str(path), "r") as f:
        try:
            out = f.read(flat, size=nbytes, file_offset=file_offset)
        except TypeError:
            try:
                out = f.read(flat, nbytes, file_offset)
            except TypeError:
                if file_offset != 0:
                    raise
                out = f.read(flat)

    if hasattr(out, "get"):
        out = out.get()
    if out is not None and int(out) != int(nbytes):
        raise IOError(f"short GDS read from {path}: got {out}, expected {nbytes}")
    return device_array


def _load_rows_gds_torch_buffer(matrix_path, shape, offset, nbytes, target):
    """Try reading directly into a torch CUDA tensor."""
    tensor = torch.empty(shape, dtype=torch.float32, device=target)
    _read_with_kvikio(matrix_path, tensor, offset, nbytes)
    return tensor


def _load_rows_gds_numba_buffer(matrix_path, shape, offset, nbytes, target):
    """Read into a Numba CUDA array, then expose it as a torch tensor."""
    from numba import cuda

    cuda.select_device(target.index or 0)
    device_array = cuda.device_array(shape, dtype=np.float32)
    _read_with_kvikio(matrix_path, device_array, offset, nbytes)
    return torch.as_tensor(device_array, device=target)


def _load_rows_gds_cupy_buffer(matrix_path, shape, offset, nbytes, target):
    """Read into a CuPy CUDA array, then expose it as a torch tensor."""
    import cupy as cp

    cp_device = cp.cuda.Device(target.index or 0)
    with cp_device:
        gpu_array = cp.empty(shape, dtype=cp.float32)
        _read_with_kvikio(matrix_path, gpu_array, offset, nbytes)
        tensor = _torch_from_cupy(gpu_array)
    return tensor


def load_matrix_rows_torch(
    matrix_path,
    row_start=0,
    row_stop=None,
    *,
    device="cuda",
    method: LoadMethod = "auto",
):
    """Load contiguous full-width rows from a benchmark matrix.

    Parameters
    ----------
    matrix_path : str or Path
        Raw ``.dat`` file path.
    row_start, row_stop : int
        Row range ``[row_start, row_stop)``.  Columns are always all columns,
        so the requested region is contiguous in the raw row-major file.
    device : str
        Target torch device.  GDS requires a CUDA device.
    method : {"auto", "gds", "cpu"}
        ``"gds"`` requires KvikIO/cuFile and raises if unavailable.
        ``"auto"`` tries GDS for CUDA targets and falls back to CPU staging.
    """
    matrix_path = Path(matrix_path)
    _validate_file_size(matrix_path)
    meta = read_matrix_meta(matrix_path)
    p, dtype = meta["p"], meta["dtype"]
    if dtype != np.dtype("float32"):
        raise ValueError(f"Only float32 benchmark matrices are supported, got {dtype}")

    if row_stop is None:
        row_stop = p
    row_start = int(row_start)
    row_stop = int(row_stop)
    if not 0 <= row_start <= row_stop <= p:
        raise ValueError(f"invalid row range [{row_start}, {row_stop}) for p={p}")

    n_rows = row_stop - row_start
    shape = (n_rows, p)
    offset = row_start * p * dtype.itemsize
    nbytes = n_rows * p * dtype.itemsize
    target = torch.device(device)

    use_gds = method == "gds" or (method == "auto" and target.type == "cuda")
    if use_gds and target.type != "cuda":
        if method == "gds":
            raise ValueError(f"GDS requires a CUDA target device, got {target}")
        use_gds = False

    if use_gds:
        errors = []
        try:
            _require_kvikio()
        except Exception as exc:
            errors.append(("kvikio", _format_exception(exc)))
        else:
            try:
                tensor = _load_rows_gds_torch_buffer(
                    matrix_path, shape, offset, nbytes, target)
                _record_load_info(method="gds", backend="torch", nbytes=nbytes,
                                  row_start=row_start, row_stop=row_stop)
                return tensor
            except Exception as exc:
                errors.append(("torch", _format_exception(exc)))
                torch.cuda.empty_cache()
            try:
                tensor = _load_rows_gds_numba_buffer(
                    matrix_path, shape, offset, nbytes, target)
                _record_load_info(method="gds", backend="numba", nbytes=nbytes,
                                  row_start=row_start, row_stop=row_stop)
                return tensor
            except Exception as exc:
                errors.append(("numba", _format_exception(exc)))
                torch.cuda.empty_cache()
            try:
                tensor = _load_rows_gds_cupy_buffer(
                    matrix_path, shape, offset, nbytes, target)
                _record_load_info(method="gds", backend="cupy", nbytes=nbytes,
                                  row_start=row_start, row_stop=row_stop)
                return tensor
            except Exception as exc:
                errors.append(("cupy", _format_exception(exc)))
                torch.cuda.empty_cache()
        if method == "gds":
            details = "; ".join(f"{name}: {detail}"
                                for name, detail in errors)
            raise RuntimeError(f"GDS load failed for all CUDA buffer backends: {details}")

    data = np.memmap(matrix_path, dtype=dtype, mode="r", shape=(p, p))
    rows = np.array(data[row_start:row_stop], dtype=np.float32, copy=True)
    tensor = torch.from_numpy(rows).to(target)
    _record_load_info(method="cpu", backend="numpy_memmap", nbytes=nbytes,
                      row_start=row_start, row_stop=row_stop)
    return tensor


def load_matrix_torch(matrix_path, *, device="cuda", method: LoadMethod = "auto"):
    """Load a full dense benchmark matrix as a torch tensor."""
    return load_matrix_rows_torch(
        matrix_path,
        row_start=0,
        row_stop=None,
        device=device,
        method=method,
    )
