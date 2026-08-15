# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clustering-agreement metrics for the SymNMF clustering pipeline.

The notebook uses :func:`adjusted_rand_index` to compare clusterings period to
period (and SymNMF vs. spherical K-means). The Rand index counts *pairs* of
instruments and is label-permutation invariant; the *adjusted* form subtracts the
agreement expected by chance (1 = identical, ~0 = unrelated, negative = worse than
random).

Both functions are built from a ``k x k`` contingency table. That same table is
the unit of work for the distributed metric: at multi-node scale each rank holds a
row-shard of instruments, builds its local contingency, and the global table is a
single ``k x k`` ``all_reduce`` -- see :func:`distributed_adjusted_rand_index`,
which the distributed runner (``run_distributed.py``) imports. There is no GPU
dependency here; the table is tiny (``k x k``) once labels are gathered.
"""
from __future__ import annotations

import numpy as np


def _contingency(a, b):
    """Contingency table n_ij of two label arrays (rows = clusters of a, cols = b)."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((ua.size, ub.size), dtype=np.int64)
    np.add.at(table, (ia, ib), 1)
    return table


def _comb2(x):
    """Pairwise count C(x, 2) = x (x - 1) / 2, elementwise."""
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def _ari_from_table(t):
    """Adjusted Rand index from a contingency table (the shared core)."""
    sum_ij = _comb2(t).sum()
    sum_a = _comb2(t.sum(axis=1)).sum()
    sum_b = _comb2(t.sum(axis=0)).sum()
    expected = sum_a * sum_b / _comb2(t.sum())
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:
        return 1.0
    return float((sum_ij - expected) / (max_index - expected))


def rand_index(a, b):
    """(Unadjusted) Rand index: fraction of instrument pairs two clusterings agree on."""
    t = _contingency(a, b)
    total = _comb2(t.sum())
    sum_ij = _comb2(t).sum()
    sum_a = _comb2(t.sum(axis=1)).sum()
    sum_b = _comb2(t.sum(axis=0)).sum()
    tp = sum_ij
    fp = sum_b - sum_ij
    fn = sum_a - sum_ij
    tn = total - tp - fp - fn
    return float((tp + tn) / total)


def adjusted_rand_index(a, b):
    """Adjusted Rand index of two label arrays. This is the metric the notebook uses
    everywhere downstream; the contingency-table construction below is the template
    for the distributed version."""
    return _ari_from_table(_contingency(a, b))


def distributed_adjusted_rand_index(labels_a, labels_b, num_clusters, group=None):
    """Multi-node ARI via a single ``k x k`` contingency ``all_reduce`` (template).

    Each rank passes the labels for *its* row-shard of instruments under two
    clusterings (``labels_a``, ``labels_b``), each in ``[0, num_clusters)``. We build
    the local contingency, sum it across ranks with one NCCL ``all_reduce`` (the table
    is only ``k x k``), and every rank computes the identical global ARI. This is the
    distributed counterpart of :func:`adjusted_rand_index`; it lives outside cuML
    (which is single-GPU) and rides the same ``torch.distributed`` / NCCL runtime as
    the AdaptGrow solver and spherical K-means.

    Not exercised by the notebook (single-GPU); imported by ``run_distributed.py``.
    """
    import torch
    import torch.distributed as dist

    a = torch.as_tensor(labels_a, dtype=torch.long)
    b = torch.as_tensor(labels_b, dtype=torch.long)
    k = int(num_clusters)
    device = a.device if a.is_cuda else (
        torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"))

    flat = (a.to(device) * k + b.to(device)).clamp_(0, k * k - 1)
    table = torch.zeros(k * k, dtype=torch.float64, device=device)
    table.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(table, op=dist.ReduceOp.SUM, group=group)

    return _ari_from_table(table.reshape(k, k).cpu().numpy())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.integers(0, 8, size=500)
    y = rng.integers(0, 8, size=500)
    assert abs(adjusted_rand_index(x, x) - 1.0) < 1e-12
    assert abs(adjusted_rand_index(x, y)) < 0.05         # unrelated -> ~0
    assert 0.0 <= rand_index(x, y) <= 1.0
    print("adjusted_rand_index self-test passed",
          f"(ARI(x,x)={adjusted_rand_index(x, x):.3f}, ARI(x,y)={adjusted_rand_index(x, y):.3f})")
