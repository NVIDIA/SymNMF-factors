# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-node / multi-GPU runner for the AdaptGrow reference implementation.

Same pipeline, same two core algorithms (AdaptGrow SymNMF + spherical K-means) — only
the launcher and the data path differ from the notebook. ``S`` is never held whole on
one GPU: it is **row-sharded** as a ``DistributedMatrix`` (GPU *i* holds rows
``[i·n/P, (i+1)·n/P)``), the solvers ride that layout over ``torch.distributed`` / NCCL,
and the ARI is a ``k×k`` ``all_reduce`` (``clustering_metrics.distributed_adjusted_rand_index``).
No Dask, no second array runtime.

SCOPE — distributed AdaptGrow factorization is implemented for prebuilt
row-sharded matrices. These end-to-end workflow pieces are not included:
  * in-job streaming estimator (build each ``S_block`` from a replicated ``R`` without first
    writing the whole ``S`` to disk) — for now we consume pre-built ``S_t`` matrix files;
  * distributed spherical K-means (a PyTorch/NCCL method on the same row-sharded layout);
  * porting ``choose_rank`` onto the sharded ``S·X`` products.

Launch with ``torchrun`` (single node) or ``srun`` (multi-node) — see
``docs/distributed.md``.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

# Reused, already-implemented building blocks (no reimplementation here).
from adaptgrow import AdaptGrow
from adaptgrow.clustering_metrics import distributed_adjusted_rand_index
from adaptgrow.distributed import DistributedMatrix, shard_matrix_file


def init_distributed():
    """Initialise the process group from the launcher's environment.

    ``torchrun`` / ``srun`` export ``RANK``, ``WORLD_SIZE`` and ``LOCAL_RANK``; we use the
    ``env://`` init method so the same code runs on one node or many. (This is the
    multi-node generalisation of ``adaptgrow.distributed.setup``, which hardcodes localhost.)
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def log0(rank, msg):
    """Print from rank 0 only."""
    if rank == 0:
        print(msg, flush=True)


def factorize_distributed(S_dist, k, H_init=None, n_iter=300, lr=2.0, seed=7):
    """Symmetric NMF ``S ≈ H Hᵀ`` with the production AdaptGrow solver against the
    row-sharded matrix. ``DistributedMatrix`` exposes a tensor-like interface
    (``@``, ``sqnorm``, ``diag``), so this is the same solver call as the notebook;
    ``H`` is replicated on every rank. Warm-started from the previous window's ``H``."""
    if H_init is None:
        torch.manual_seed(seed)
    return AdaptGrow(lr=lr, entry_frac=1.0, max_iter=n_iter,
                   grad_tol=1e-4, verbose=False).optimize(S_dist, k, H0=H_init)


def spherical_kmeans_distributed(S_dist, k, centroids=None):
    """Distributed spherical K-means on the same row-sharded layout: row-local
    assignments, a ``k×n`` centroid ``all_reduce`` each iteration.

    Distributed spherical K-means is not included in this release. The
    single-GPU reference is ``spherical_kmeans`` in the notebook."""
    raise NotImplementedError("distributed spherical K-means: pending (see README.md)")


def window_matrix_paths(matrix_dir, kind, suffix):
    """Sorted list of pre-built per-window ``S_t`` matrix files for a given kind
    ('corr' or 'tpdm'). Built offline by corr_construction.py / tpdm_construction.py
    (tile-by-tile to a memmap), which is the current estimator path at scale.

    In-job streaming matrix construction is not included in this release."""
    files = sorted(
        f for f in os.listdir(matrix_dir) if f.startswith(kind) and f.endswith(suffix)
    )
    return [os.path.join(matrix_dir, f) for f in files]


def run(args):
    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank)
    log0(rank, f"world_size={world_size}  device={device}  matrix_dir={args.matrix_dir}")

    paths = window_matrix_paths(args.matrix_dir, args.kind, args.suffix)
    log0(rank, f"{len(paths)} window matrices of kind '{args.kind}'")

    H_prev, labels_prev = None, None
    aris = []
    t0 = time.perf_counter()
    for t, path in enumerate(paths):
        # Step: shard this window's S across ranks (each rank reads only its rows).
        S_dist = shard_matrix_file(path, rank, world_size, device, method=args.io_method)

        # Step: factorize (warm-started); H is replicated, so labels are global.
        H = factorize_distributed(S_dist, args.k, H_init=H_prev, n_iter=args.n_iter)
        labels = H.argmax(1)
        H_prev = H

        # Step: stability via ARI against the previous window. distributed_adjusted_rand_index
        # reduces a k×k contingency across ranks, so it is correct whether labels are global
        # (replicated H) or row-local (the sharded K-means assignments, once wired).
        if labels_prev is not None:
            ari_t = distributed_adjusted_rand_index(labels_prev, labels, args.k)
            aris.append(ari_t)
            log0(rank, f"  window {t:4d}: ARI(t-1,t) = {ari_t:.3f}")
        labels_prev = labels
        del S_dist

    if rank == 0:
        torch.cuda.synchronize()
        print(f"factored {len(paths)} windows at n (sharded over {world_size} GPUs) "
              f"in {time.perf_counter() - t0:.1f}s; mean ARI = "
              f"{sum(aris) / len(aris):.3f}" if aris else "no ARI computed")

    dist.destroy_process_group()


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix-dir", required=True,
                   help="directory of pre-built per-window S_t matrix files")
    p.add_argument("--kind", default="corr", choices=["corr", "tpdm"])
    p.add_argument("--suffix", default=".dat", help="matrix-file suffix")
    p.add_argument(
        "--k",
        type=int,
        default=8,
        help="factorization rank (distributed rank selection is not included)",
    )
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--io-method", default="auto", choices=["auto", "gds", "cpu"])
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
