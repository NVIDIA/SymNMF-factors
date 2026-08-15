# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Construct synthetic Tail Pairwise Dependence Matrices (TPDM).

Generates X = A @ Z where A is a block-structured non-negative mixing
matrix and Z has iid Pareto(α=2) entries.  Applies polar transform,
filters to the top quantile by radial magnitude, and computes the
empirical TPDM: Σ̂ = (p / n_exc) W^T W.

By Corollary A.2 of Ghita et al., arXiv:2607.24518, the theoretical TPDM
is Σ = A A^T.  The empirical estimator converges to this as n_sample → ∞.

Scalability
-----------
At p <= 10k the generator runs entirely in RAM.  At p = 100k the
p × n_sample intermediates (~16 GB each at n_sample=40k) are generated
in row-chunks to cap peak memory.  At p = 1M the output (4 TB) is
written tile-by-tile to a numpy memmap on disk.

Interface:
    Sigma, perm, labels = tpdm_construct(p, random_state=42)
    # Sigma is already in permuted order — sequential reads are efficient.
    # perm maps permuted index → block-ordered index.
    # labels[i] is the ground-truth cluster of row i.
    # For large p:
    Sigma, perm, labels = tpdm_construct(p, random_state=42, output_path="tpdm.dat")
"""

import warnings

import numpy as np

from .benchmark_loader import _save_companion_files


# ── internal helpers ──────────────────────────────────────────────────

def _generate_pareto_matrix(rng, k, n_sample, alpha=2):
    """Pareto Type I samples, shape (k, n_sample)."""
    return (rng.pareto(alpha, (k, n_sample)) + 1).astype(np.float32)


def _random_block_sizes(rng, p, k, jitter=0.40):
    """Return k+1 boundary indices partitioning [0, p) into k blocks
    whose sizes vary by up to ±jitter around the mean size p/k.

    Every block gets at least 1 row.
    """
    mean_size = p / k
    raw = mean_size * (1.0 + jitter * (2 * rng.random(k) - 1))
    raw = np.maximum(raw, 1.0)
    raw = raw / raw.sum() * p
    sizes = np.round(raw).astype(int)
    sizes = np.maximum(sizes, 1)

    # Fix rounding residual by adjusting the largest block
    diff = p - sizes.sum()
    sizes[np.argmax(sizes)] += diff

    starts = np.empty(k + 1, dtype=int)
    starts[0] = 0
    np.cumsum(sizes, out=starts[1:])
    return starts


def _generate_block_mixing_matrix(rng, p, k, noise_level=0.05,
                                   common_loading=0.8, cluster_loading=1.2):
    """Block-structured p×(k+1) non-negative mixing matrix.

    Column 0 is a common factor that loads on all instruments (captures
    shared variation present across any asset class — e.g. broad market
    moves in equities, or rate-level shifts in a derivatives book).
    Columns 1..k are cluster factors, each with a contiguous block of
    strong entries and small off-block noise.  These represent latent
    groups that emerge empirically (co-dependent strikes, correlated
    credit names, geographically similar assets, etc.).

    Block sizes vary by ±40% around p/k (largest/smallest ≈ 2–3:1)
    to reflect realistic group-size heterogeneity.

    Default calibration: common-factor R² ≈ 20%, cluster R² ≈ 45%,
    idiosyncratic ≈ 35%.

    Returns
    -------
    M : ndarray (p, k+1), float32
        Non-negative mixing matrix.
    labels : ndarray (p,), int32
        Ground-truth cluster label for each row (0..k-1).
    """
    if p < k:
        warnings.warn(f"p={p} < k={k}: mixing matrix cannot have full column rank.")

    weights = np.zeros((p, k), dtype=np.float32)
    row_starts = _random_block_sizes(rng, p, k) if p >= k else \
        np.linspace(0, p, k + 1, dtype=int)

    labels = np.empty(p, dtype=np.int32)
    for col in range(k):
        weights[row_starts[col]:row_starts[col + 1], col] = 1.0
        labels[row_starts[col]:row_starts[col + 1]] = col

    block_content = rng.random((p, k)) * cluster_loading + cluster_loading * 0.5
    clusters = block_content * weights + noise_level * rng.standard_normal((p, k))

    common = (common_loading + 0.15 * rng.standard_normal(p)).reshape(-1, 1)

    M = np.hstack([common, clusters])
    return np.maximum(M, 0.0).astype(np.float32), labels


def _to_polar(M):
    """Column-wise polar decomposition → (norms, unit directions)."""
    norms = np.linalg.norm(M, axis=0)
    return norms, M / norms


def _top_n_directions(norms, directions, n_exc):
    """Select the n_exc columns with largest norms; return their unit directions."""
    idx = np.argpartition(norms, -n_exc)[-n_exc:]
    return directions[:, idx]


# ── chunked generation for large p ────────────────────────────────────

_CHUNK_ROWS = 10_000  # row-block size for chunked generation


def _generate_M_chunked(rng, A, Z, p, n_sample, chunk_rows=_CHUNK_ROWS):
    """Generate M = A @ Z + eps in row-chunks, returning column norms
    without ever materialising the full p × n_sample array.

    A second pass via _extract_W_chunked uses the norms + exceedance
    indices to assemble the angular exceedance matrix W.
    """
    norms_sq = np.zeros(n_sample, dtype=np.float64)

    for r0 in range(0, p, chunk_rows):
        r1 = min(r0 + chunk_rows, p)
        A_block = A[r0:r1]
        eps_block = 0.3 * (rng.pareto(2, (r1 - r0, n_sample)) + 1).astype(np.float32)
        M_block = A_block @ Z + eps_block
        norms_sq += np.sum(M_block.astype(np.float64) ** 2, axis=0)
        del eps_block, M_block

    norms = np.sqrt(norms_sq).astype(np.float32)
    return norms


def _extract_W_chunked(rng_fresh, A, Z, p, n_sample, norms, exc_idx,
                       chunk_rows=_CHUNK_ROWS):
    """Second pass: regenerate M row-by-row and extract W[:, exc_idx].

    rng_fresh must be a fresh RNG seeded to the same state as the
    first pass's eps generation, so that the same eps is produced.
    """
    n_exc = len(exc_idx)
    inv_norms = (1.0 / norms[exc_idx]).astype(np.float32)
    W = np.empty((p, n_exc), dtype=np.float32)

    for r0 in range(0, p, chunk_rows):
        r1 = min(r0 + chunk_rows, p)
        A_block = A[r0:r1]
        eps_block = 0.3 * (rng_fresh.pareto(2, (r1 - r0, n_sample)) + 1).astype(np.float32)
        M_block = A_block @ Z + eps_block
        W[r0:r1] = M_block[:, exc_idx] * inv_norms
        del eps_block, M_block

    return W


# ── output helpers ────────────────────────────────────────────────────

def _allocate_output(p, output_path):
    """Return a (p, p) fp32 array — in-memory or memory-mapped."""
    if output_path is not None:
        return np.memmap(output_path, dtype=np.float32, mode='w+',
                         shape=(p, p))
    return np.empty((p, p), dtype=np.float32)


def _tiled_WWT(W, scale, Sigma, tile=_CHUNK_ROWS):
    """Compute Sigma = scale * W @ W.T tile-by-tile into a pre-allocated array."""
    p = W.shape[0]
    for i0 in range(0, p, tile):
        i1 = min(i0 + tile, p)
        for j0 in range(0, p, tile):
            j1 = min(j0 + tile, p)
            Sigma[i0:i1, j0:j1] = scale * (W[i0:i1] @ W[j0:j1].T)


def _normalise_unit_diagonal(Sigma, tile=_CHUNK_ROWS):
    """Normalise Sigma to unit diagonal, tile-by-tile for large arrays."""
    p = Sigma.shape[0]
    d = np.sqrt(np.array([Sigma[i, i] for i in range(p)], dtype=np.float32))
    d[d == 0] = 1.0

    for i0 in range(0, p, tile):
        i1 = min(i0 + tile, p)
        for j0 in range(0, p, tile):
            j1 = min(j0 + tile, p)
            Sigma[i0:i1, j0:j1] /= np.outer(d[i0:i1], d[j0:j1])

    for i in range(p):
        Sigma[i, i] = 1.0


# ── public API ────────────────────────────────────────────────────────

def tpdm_construct(p, n_sample=40000, qt_ext=0.01, permute=True,
                   random_state=None, output_path=None, chunk_rows=_CHUNK_ROWS):
    """Construct a synthetic Tail Pairwise Dependence Matrix.

    Follows Ghita et al., arXiv:2607.24518 (Corollary A.2): X = A Z
    where Z has iid Pareto(2) entries.  The theoretical TPDM is
    Σ = A A^T.  The empirical estimator is Σ̂ = (p / n_exc) W^T W.

    When ``permute=True`` (the default), the rows of the mixing matrix
    A are shuffled *before* computing Sigma, so the output is written
    directly in permuted order — no expensive post-hoc permutation of
    the p×p matrix is needed.  This makes sequential reads (row shards,
    tiles) I/O-efficient even for multi-TB outputs on memmap.

    Parameters
    ----------
    p : int
        Dimension (number of instruments / rows-cols of output).
    n_sample : int
        Number of synthetic observations. Default 40,000 corresponds to
        10 years of half-hourly snapshots (250 days/yr × 8 h/day × 2/h).
    qt_ext : float
        Fraction of observations kept as extremes (top quantile by norm).
    permute : bool
        If True (default), randomly permute A's rows before computing
        Sigma — the output matrix is already in permuted order with
        block structure hidden.  Set to False for a block-ordered
        output (debug / visual inspection).
    random_state : int or None
        Seed for full reproducibility (controls all internal RNGs).
    output_path : str or None
        If set, write the p×p result as a numpy memmap file at this path
        instead of returning an in-memory array.  Required for p >= ~100k.
    chunk_rows : int
        Row-block size for chunked generation (default 10,000).

    Returns
    -------
    Sigma : ndarray or memmap (p, p), float32
        Symmetric non-negative matrix with unit diagonal.  When
        ``permute=True``, the matrix is already in permuted order —
        solvers can read it directly with no index indirection.
    perm : ndarray or None
        Permutation vector: ``perm[i]`` is the block-ordered index
        of the instrument at permuted position ``i``.  None when
        ``permute=False``.
    block_labels : ndarray (p,), int32
        Ground-truth cluster label for each row.  When
        ``permute=True``, labels are in permuted order (matching
        Sigma's row ordering).
    """
    rng = np.random.default_rng(random_state)

    n_exc = int(n_sample * qt_ext)
    k = int(np.floor(np.sqrt(p)))

    A, block_labels = _generate_block_mixing_matrix(rng, p, k)  # p × (k+1)

    # Permute A rows early so the output Sigma is written directly in
    # permuted order — avoids an O(p²) random-access shuffle on the
    # final matrix (critical at p >= 100k where Sigma is a multi-GB/TB
    # memmap).
    perm = None
    if permute:
        perm = rng.permutation(p)
        A = A[perm]
        block_labels = block_labels[perm]

    Z = _generate_pareto_matrix(rng, k + 1, n_sample)  # (k+1) × n_sample

    use_chunked = p > chunk_rows

    if use_chunked:
        rng_state_before_eps = rng.bit_generator.state

        norms = _generate_M_chunked(rng, A, Z, p, n_sample, chunk_rows)

        exc_idx = np.argpartition(norms, -n_exc)[-n_exc:]

        rng_pass2 = np.random.default_rng(0)
        rng_pass2.bit_generator.state = rng_state_before_eps
        W = _extract_W_chunked(rng_pass2, A, Z, p, n_sample, norms,
                               exc_idx, chunk_rows)

    else:
        eps = 0.3 * (rng.pareto(2, (p, n_sample)) + 1).astype(np.float32)
        M = A @ Z + eps
        del eps

        norms, directions = _to_polar(M)
        del M
        W = _top_n_directions(norms, directions, n_exc)
        del directions

    del Z, A

    scale = np.float32(p / n_exc)
    Sigma = _allocate_output(p, output_path)
    _tiled_WWT(W, scale, Sigma, tile=chunk_rows)
    del W

    _normalise_unit_diagonal(Sigma, tile=chunk_rows)

    if output_path is not None:
        Sigma.flush()
        _save_companion_files(output_path, p, perm, block_labels)

    return Sigma, perm, block_labels


if __name__ == "__main__":
    p = 6

    # Default: permuted output (what solvers see)
    Sigma, perm, labels = tpdm_construct(p, random_state=42)
    print("Permuted TPDM (default):\n", Sigma)
    print("Permutation (permuted→block-ordered):", perm)
    print("Cluster labels:", labels)

    # Debug: block-ordered output
    Sigma_bo, _, labels_bo = tpdm_construct(p, random_state=42, permute=False)
    print("\nBlock-ordered TPDM (debug):\n", Sigma_bo)
    print("Cluster labels (block-ordered):", labels_bo)
