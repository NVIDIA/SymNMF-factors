# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark matrix loader for SymNMF experiments.

Generators (tpdm_construct, corr_construct) now write matrices directly
in permuted order — the file on disk is what solvers see.  Sequential
reads (row shards, tiles) are I/O-efficient with no index indirection.

The permutation vector, saved alongside the matrix, maps each permuted
index back to its block-ordered position — useful only for recovering
ground-truth cluster labels, not for read-time reordering.

Usage
-----
    from adaptgrow.benchmark_loader import BenchmarkMatrix
    from adaptgrow.tpdm_construction import tpdm_construct

    # Generate (output is already permuted)
    Sigma, perm, labels = tpdm_construct(p=1000, random_state=42)
    mat = BenchmarkMatrix(Sigma, perm, labels)
    shard = mat.load_rows(0, 256)         # sequential read, no indirection
    tile  = mat.load_tile(0, 256, 0, 256) # contiguous tile
    A     = mat.to_dense()                # full matrix (small p)

    # Save to disk and reload (large p)
    mat.save("tpdm_p1000.dat")
    mat2 = BenchmarkMatrix.from_files("tpdm_p1000.dat")
    assert np.allclose(mat.to_dense(), mat2.to_dense())

    # Ground-truth cluster labels (saved with the matrix)
    print(mat2.labels)  # label per row, matching Sigma's row order
"""

import json
import os

import numpy as np


class BenchmarkMatrix:
    """Loader for dense symmetric benchmark matrices.

    The underlying data is stored in permuted order (as the generators
    produce it).  All read methods return data directly — no permutation
    indirection.

    The permutation vector ``perm`` is kept for ground-truth recovery:
    ``perm[i]`` is the block-ordered index of the instrument at
    permuted position ``i``.
    """

    def __init__(self, data, perm=None, labels=None, matrix_path=None):
        """
        Parameters
        ----------
        data : ndarray or memmap, shape (p, p), float32
            Symmetric matrix in permuted order (ready for solvers).
        perm : ndarray of int, length p, or None
            Permutation vector: ``perm[i]`` gives the block-ordered
            index of permuted row/col ``i``.
        labels : ndarray of int, length p, or None
            Ground-truth cluster label for each row, in the same order
            as the rows of ``data``.
        """
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {data.shape}")
        self._data = data
        self._perm = perm
        self._labels = labels
        self._matrix_path = matrix_path
        self.p = data.shape[0]

    # ── constructors ──────────────────────────────────────────────────

    @classmethod
    def from_files(cls, matrix_path):
        """Load matrix + permutation from disk.

        Expects companion files alongside the ``.dat`` memmap:

        - ``matrix_path + ".meta.json"``  — ``{"p": int, "dtype": str}``
        - ``matrix_path + ".perm.npy"``   — permutation vector (optional)
        """
        meta_path = matrix_path + ".meta.json"
        perm_path = matrix_path + ".perm.npy"

        with open(meta_path) as f:
            meta = json.load(f)
        p = meta["p"]
        dtype = np.dtype(meta.get("dtype", "float32"))

        labels_path = matrix_path + ".labels.npy"

        data = np.memmap(matrix_path, dtype=dtype, mode="r", shape=(p, p))
        perm = np.load(perm_path) if os.path.exists(perm_path) else None
        labels = np.load(labels_path) if os.path.exists(labels_path) else None
        return cls(data, perm, labels, matrix_path=matrix_path)

    # ── persistence ───────────────────────────────────────────────────

    def save(self, matrix_path):
        """Write matrix + companion files to disk.

        For in-memory data a new memmap file is created.  For data that
        already lives on disk (memmap) only the companion files are
        written.
        """
        if not isinstance(self._data, np.memmap):
            fp = np.memmap(matrix_path, dtype=self._data.dtype,
                           mode="w+", shape=self._data.shape)
            fp[:] = self._data
            fp.flush()

        _save_companion_files(matrix_path, self.p, self._perm,
                              self._labels)

    # ── direct read access (no permutation indirection) ───────────────

    def load_rows(self, i0, i1):
        """Rows ``[i0, i1)`` → ``(i1-i0, p)`` array.  Sequential read."""
        return np.asarray(self._data[i0:i1])

    def load_rows_torch(self, i0, i1, *, device="cuda", method="auto"):
        """Rows ``[i0, i1)`` as a torch tensor.

        For file-backed matrices, ``method="gds"`` uses the GDS loader from
        ``matrix_io`` and raises if KvikIO/cuFile is unavailable.  ``"auto"``
        tries GDS on CUDA targets and falls back to CPU staging.
        """
        if self._matrix_path is not None:
            from .matrix_io import load_matrix_rows_torch
            return load_matrix_rows_torch(
                self._matrix_path, i0, i1, device=device, method=method)

        import torch

        rows = np.array(self.load_rows(i0, i1), dtype=np.float32, copy=True)
        return torch.from_numpy(rows).to(device)

    def to_torch(self, *, device="cuda", method="auto"):
        """Full matrix as a torch tensor.  Practical only when memory allows."""
        return self.load_rows_torch(0, self.p, device=device, method=method)

    def load_tile(self, i0, i1, j0, j1):
        """Tile ``[i0:i1, j0:j1)`` → ``(i1-i0, j1-j0)`` array."""
        return np.asarray(self._data[i0:i1, j0:j1])

    def to_dense(self):
        """Full matrix in RAM.  Only practical for p < ~50k."""
        return np.asarray(self._data)

    # ── ground-truth recovery ─────────────────────────────────────────

    @property
    def labels(self):
        """Ground-truth cluster labels, or ``None``."""
        return self._labels

    # ── properties ────────────────────────────────────────────────────

    @property
    def shape(self):
        return (self.p, self.p)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def data(self):
        """Direct access to underlying storage."""
        return self._data

    @property
    def matrix_path(self):
        """Backing ``.dat`` path for file-backed matrices, or ``None``."""
        return self._matrix_path

    @property
    def perm(self):
        """Permutation vector, or ``None``."""
        return self._perm


# ── utility shared with generators ────────────────────────────────────

def _save_companion_files(output_path, p, perm=None, labels=None):
    """Write companion files alongside a memmap file.

    Files written:
    - ``.meta.json``   — ``{"p": int, "dtype": str}``
    - ``.perm.npy``    — permutation vector (if provided)
    - ``.labels.npy``  — ground-truth cluster labels (if provided)
    """
    with open(output_path + ".meta.json", "w") as f:
        json.dump({"p": int(p), "dtype": "float32"}, f)
    if perm is not None:
        np.save(output_path + ".perm.npy", perm)
    if labels is not None:
        np.save(output_path + ".labels.npy", labels)
