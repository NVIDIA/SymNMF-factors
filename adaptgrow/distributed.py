# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed SymNMF: row-sharded A across GPUs on a single node.

Architecture
------------
- A is row-sharded: GPU i holds rows [i*n/P, (i+1)*n/P) of A (all n columns)
- H is replicated across all GPUs (identical on every rank)
- Communication per iteration:
    Forward:  all_gather of n×k   (local A_i @ H pieces → full AH)
    Backward: all_reduce of n×k   (partial grads → full grad_H)
- Because all GPUs receive identical AH and compute identical gradients,
  optimizer state stays in sync without additional communication.

Usage
-----
    from adaptgrow.distributed import DistributedMatrix, cleanup, setup

    setup(rank, world_size)
    A_dist = DistributedMatrix(A_local, n_global)
    H = AdaptGrow(lr=2.0).optimize(A_dist, k)
    cleanup()

Reference: this 1D row-sharding is the symmetric specialisation of the
MPI-FAUN framework (Kannan, Ballard, Park, 2016, arXiv:1609.09154).
For square symmetric A with a single factor H, the 2D processor grid
degenerates to a 1D row partition.
"""

import os
import numpy as np
import scipy.sparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ───────────────────────────── process group helpers ─────────────────────────

def setup(rank, world_size, backend="nccl", port="12355"):
    """Initialise the distributed process group for a single-node run."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


# ───────────────────── autograd-aware distributed matmul ─────────────────────

class _DistributedMatmul(torch.autograd.Function):
    """A @ H where A is row-sharded across GPUs.

    Forward:  local A_local @ H  →  all_gather  →  full AH  (n × k)
    Backward: A_local^T @ ∂L/∂(AH)_local  →  all_reduce(SUM)  →  full ∂L/∂H
    """

    @staticmethod
    def forward(ctx, A_local, H):
        ctx.A_local = A_local
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()

        local_AH = A_local @ H                    # (n/P) × k  or  (n/P) for 1-D H
        gathered = [torch.empty_like(local_AH) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, local_AH.contiguous())
        return torch.cat(gathered, dim=0)          # n × k

    @staticmethod
    def backward(ctx, grad_output):
        A_local = ctx.A_local
        n_local = A_local.shape[0]
        start = ctx.rank * n_local
        end = start + n_local

        grad_local = grad_output[start:end]
        if A_local.is_sparse_csr or A_local.is_sparse:
            grad_H = A_local.t() @ grad_local.contiguous()
        else:
            grad_H = A_local.T @ grad_local.contiguous()
        dist.all_reduce(grad_H, op=dist.ReduceOp.SUM)
        return None, grad_H


# ───────────────────────── DistributedMatrix wrapper ─────────────────────────

class DistributedMatrix:
    """Row-sharded symmetric matrix for distributed SymNMF.

    Provides the tensor-like interface consumed by ``AdaptGrow``.

    Supported operations
    --------------------
    A @ H             autograd-aware distributed matmul  (__matmul__)
    other @ A         distributed rmatmul, no autograd   (__rmatmul__)
    A.sqnorm()        ||A||_F^2 via all_reduce           (cached)
    A.diag()          diagonal extraction via all_gather  (cached)
    A.full()          reconstruct full n×n tensor         (one-shot)
    A.size(dim)       global n
    A.device / dtype  local GPU device / dtype
    """

    def __init__(self, A_local, n_global):
        ws = dist.get_world_size() if dist.is_initialized() else 1
        if n_global % ws != 0:
            raise ValueError(
                f"n={n_global} must be divisible by world_size={ws}")
        self.data = A_local          # (n/P) × n  on this GPU
        self._n = n_global
        self._sqnorm_cache = None
        self._diag_cache = None

    # ── metadata (tensor-like) ──────────────────────────────────────────

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    def size(self, dim=None):
        if dim is None:
            return torch.Size([self._n, self._n])
        return self._n

    @property
    def shape(self):
        return torch.Size([self._n, self._n])

    # ── core distributed operations ─────────────────────────────────────

    def __matmul__(self, H):
        """A @ H  →  n × k   (with autograd)."""
        return _DistributedMatmul.apply(self.data, H)

    def __rmatmul__(self, other):
        """other @ A  (no autograd — used by Randomized EVD's Q^T A Q).

        other:  l × n   →   result:  l × n
        Each GPU computes other[:, start:end] @ A_local  then  all_reduce.
        """
        n_local = self.data.shape[0]
        rank = dist.get_rank()
        start = rank * n_local
        end = start + n_local

        partial = other[:, start:end].contiguous() @ self.data   # l × n
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial

    # ── scalar / vector queries ─────────────────────────────────────────

    def sqnorm(self):
        """||A||_F^2  via local sum + all_reduce.  Cached after first call."""
        if self._sqnorm_cache is None:
            if self.data.is_sparse_csr:
                local_sq = self.data.values().pow(2).sum()
            elif self.data.is_sparse:
                local_sq = self.data._values().pow(2).sum()
            else:
                local_sq = torch.sum(self.data * self.data)
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
            self._sqnorm_cache = local_sq.detach()
        return self._sqnorm_cache

    def diag(self):
        """Diagonal of A.  Used by Newton's Hessian approximation.  Cached."""
        if self._diag_cache is None:
            n_local = self.data.shape[0]
            rank = dist.get_rank()
            start = rank * n_local
            dev, dt = self.data.device, self.data.dtype

            if self.data.is_sparse_csr:
                crow = self.data.crow_indices()
                col = self.data.col_indices()
                vals = self.data.values()
                local_diag = torch.zeros(n_local, device=dev, dtype=dt)
                for i in range(n_local):
                    s, e = crow[i].item(), crow[i + 1].item()
                    global_col = start + i
                    cols_i = col[s:e]
                    mask = cols_i == global_col
                    if mask.any():
                        local_diag[i] = vals[s:e][mask].sum()
            else:
                local_idx = torch.arange(n_local, device=dev)
                global_idx = local_idx + start
                local_diag = self.data[local_idx, global_idx]

            gathered = [torch.empty(n_local, device=dev, dtype=dt)
                        for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local_diag.contiguous())
            self._diag_cache = torch.cat(gathered)
        return self._diag_cache

    def full(self):
        """Gather the full n×n matrix on every rank.

        Needed once for NNDSVD initialisation.  All ranks must call this.
        For sparse shards, converts to dense before gathering.
        """
        if self.data.is_sparse_csr or self.data.is_sparse:
            local_dense = self.data.to_dense()
        else:
            local_dense = self.data
        gathered = [torch.empty_like(local_dense)
                    for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_dense.contiguous())
        return torch.cat(gathered, dim=0)

    # ── sampled (stochastic) operations ─────────────────────────────────
    #
    # The stochastic solvers sample row sets I (and column sets J) of A.
    # Because A is row-sharded, every sampled row lives on exactly one rank,
    # and SymNMF's stochastic gradient is row-separable: row i only needs
    # A[i] (its owner has it) and the replicated H.  So each rank computes
    # the contribution of the sampled rows it OWNS, and an all_reduce(SUM)
    # combines them into the identical full result on every rank — keeping
    # H / G replicated with no extra communication beyond the reduce.
    #
    # Requires `idx`/`I`/`J` to be identical across ranks (guaranteed when
    # every rank runs the solver with the same RNG seed/state).

    def _owned_rows(self, idx):
        """Return (mask, local_offsets) for the sampled global indices this
        rank owns. `idx` must be a 1-D LongTensor of global row indices."""
        n_local = self.data.shape[0]
        rank = dist.get_rank() if dist.is_initialized() else 0
        start = rank * n_local
        mask = (idx >= start) & (idx < start + n_local)
        return mask, idx[mask] - start

    def index_matmul(self, idx, H):
        """``A[idx] @ H`` (|idx| × k), identical on every rank.

        Owner-computes: rank fills only the sampled rows it holds; the
        all_reduce(SUM) sums the disjoint contributions (each sampled row is
        owned by exactly one rank). Communication is O(|idx|·k).
        """
        out = torch.zeros(idx.shape[0], H.shape[1],
                          device=H.device, dtype=H.dtype)
        mask, local_rows = self._owned_rows(idx)
        if local_rows.numel() > 0:
            out[mask] = self.data.index_select(0, local_rows) @ H
        if dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def block_entry_grad(self, H, I, J, chunk=4096):
        """Distributed block-entry stochastic gradient (full n×k), identical
        on every rank:

            R_IJ      = H_I H_J^T - A_IJ
            grad[I]  += 2 R_IJ   H_J     (rows I, disjoint by owner)
            grad[J]  += 2 R_IJ^T H_I     (rows J, summed over I-owners)

        Each rank only iterates the I-rows it owns (slicing columns J from its
        local rows), then an all_reduce(SUM) over the affected rows (I ∪ J)
        combines: grad[I] rows are disjoint by owner, grad[J] is the sum of
        per-owner partials. Communication is O(|I ∪ J|·k).
        """
        grad = torch.zeros_like(H)
        H_J = H[J]
        mask, I_local = self._owned_rows(I)
        I_owned = I[mask]                       # global indices this rank owns
        if I_local.numel() > 0:
            H_Iown = H[I_owned]
            A_rows = self.data.index_select(0, I_local)      # (|I_p|, n) local
            bs_i, bs_j = I_owned.numel(), J.numel()
            for i0 in range(0, bs_i, chunk):
                I_c = I_owned[i0:i0 + chunk]
                H_Ic = H_Iown[i0:i0 + chunk]
                A_rc = A_rows[i0:i0 + chunk]
                for j0 in range(0, bs_j, chunk):
                    J_c = J[j0:j0 + chunk]
                    H_Jc = H_J[j0:j0 + chunk]
                    A_IJ = A_rc.index_select(1, J_c)
                    R = torch.addmm(A_IJ, H_Ic, H_Jc.T, beta=-1.0, alpha=1.0)
                    grad.index_add_(0, I_c, R @ H_Jc, alpha=2.0)
                    grad.index_add_(0, J_c, R.T @ H_Ic, alpha=2.0)
        if dist.is_initialized():
            affected = torch.unique(torch.cat([I, J]))
            g = grad.index_select(0, affected).contiguous()
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            grad.index_copy_(0, affected, g)
        return grad


# ───────────────────────── convenience: shard from numpy / scipy ──────────────

def _scipy_csr_rows_to_torch(sp_csr, row_start, row_end, device, dtype):
    """Slice rows from a scipy CSR matrix and return a torch.sparse_csr_tensor."""
    sub = sp_csr[row_start:row_end].tocsr()
    crow = torch.tensor(sub.indptr, dtype=torch.int64, device=device)
    col = torch.tensor(sub.indices, dtype=torch.int64, device=device)
    vals = torch.tensor(sub.data, dtype=dtype, device=device)
    return torch.sparse_csr_tensor(crow, col, vals,
                                   size=(row_end - row_start, sp_csr.shape[1]),
                                   device=device)


def shard_matrix(A, rank, world_size, device, as_sparse=False):
    """Build a DistributedMatrix from a full matrix (dense ndarray or
    scipy.sparse).  Each rank keeps only its row shard on GPU.

    Parameters
    ----------
    A : ndarray (n, n) or scipy.sparse matrix
    rank : int
    world_size : int
    device : torch device
    as_sparse : bool
        If True and A is dense, sparsify the shard into CSR on GPU.
        If A is already scipy.sparse, shards are always CSR.
    """
    n = A.shape[0]
    n_local = n // world_size
    r_start = rank * n_local
    r_end = r_start + n_local

    if scipy.sparse.issparse(A):
        A_csr = A.tocsr()
        A_local = _scipy_csr_rows_to_torch(A_csr, r_start, r_end,
                                           device, torch.float32)
    elif as_sparse:
        shard_np = A[r_start:r_end]
        sp_shard = scipy.sparse.csr_matrix(shard_np.astype(np.float32))
        A_local = _scipy_csr_rows_to_torch(
            sp_shard, 0, n_local, device, torch.float32)
    else:
        A_full = torch.tensor(A, dtype=torch.float32)
        A_local = A_full[r_start:r_end].to(device)

    return DistributedMatrix(A_local, n)


def shard_matrix_file(matrix_path, rank, world_size, device, method="auto"):
    """Build a ``DistributedMatrix`` by reading this rank's rows from disk.

    ``method="gds"`` requires KvikIO/cuFile and loads the row shard directly
    into GPU memory.  ``method="auto"`` tries GDS for CUDA targets and falls
    back to CPU staging if the GDS Python stack is unavailable.
    """
    from .matrix_io import load_matrix_rows_torch, read_matrix_meta

    n = read_matrix_meta(matrix_path)["p"]
    if n % world_size != 0:
        raise ValueError(f"n={n} must be divisible by world_size={world_size}")

    n_local = n // world_size
    r_start = rank * n_local
    r_end = r_start + n_local
    A_local = load_matrix_rows_torch(
        matrix_path, r_start, r_end, device=device, method=method)
    return DistributedMatrix(A_local, n)
