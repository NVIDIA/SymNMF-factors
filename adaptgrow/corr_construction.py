# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Construct correlation matrices from time-series data,
with non-negative transformations suitable for SymNMF.

Parallel to tpdm_construction.py:
  - TPDM: X = A @ Z with Pareto Z, takes extremes → angular dependence
  - Corr: X = A @ Z with Gaussian Z, full sample → linear dependence
Both share the same block-structured mixing matrix A, producing symmetric
matrices with unit diagonal and values in [0, 1].  The construction matches
Ghita et al., arXiv:2607.24518.

Scalability
-----------
At p <= 10k the generator runs entirely in RAM.  At p = 100k the
p × n_sample intermediates (~16 GB each at n_sample=40k) are generated
in row-chunks and the covariance is accumulated tile-by-tile.  At
p = 1M the output (4 TB) is written tile-by-tile to a numpy memmap.

Interface:
    Sigma, perm, labels = corr_construct(p=100, random_state=42)
    # Sigma is already in permuted order — sequential reads are efficient.
    # perm maps permuted index → block-ordered index.
    # labels[i] is the ground-truth cluster of row i.
    # For large p:
    Sigma, perm, labels = corr_construct(p=100000, random_state=42,
                                         output_path="corr.dat")
"""

import numpy as np

from .benchmark_loader import _save_companion_files
from .tpdm_construction import (_generate_block_mixing_matrix,
                                _allocate_output, _normalise_unit_diagonal,
                                _CHUNK_ROWS)


# ── estimators ────────────────────────────────────────────────────────

def sample_correlation(X):
    """Pearson correlation.  X is (p, n_sample).  Uses n-1 denominator."""
    return np.corrcoef(X)


def shrunk_correlation(X):
    """Ledoit-Wolf linear shrinkage toward scaled identity, converted to
    correlation.

    Uses unbiased covariance (n-1 denominator), consistent with
    np.corrcoef and the original formulation in Ledoit & Wolf (2004,
    "A well-conditioned estimator for large-dimensional covariance
    matrices", J. Multivariate Analysis 88(2), 365-411).

    Returns (C, alpha) where alpha is the shrinkage intensity in [0, 1].
    """
    p, n = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    S = (Xc @ Xc.T) / (n - 1)

    mu = np.trace(S) / p
    delta = np.sum((S - mu * np.eye(p)) ** 2) / p

    X2 = Xc ** 2
    beta_hat = np.sum(X2 @ X2.T / (n - 1) - S ** 2) / (p * (n - 1))
    alpha = min(max(beta_hat / delta, 0.0), 1.0)

    S_shrunk = (1 - alpha) * S + alpha * mu * np.eye(p)

    d = np.sqrt(np.diag(S_shrunk))
    d[d == 0] = 1.0
    C = S_shrunk / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C, alpha


# ── non-negative transformations ──────────────────────────────────────

def absolute(C):
    """Element-wise |C|.  Anti-correlated pairs stay strong."""
    return np.abs(C)


def squared(C):
    """Element-wise C².  Suppresses weak correlations harder.
    Preserves PSD (Schur product theorem).
    """
    return C ** 2


def positive_part(C):
    """max(0, C).  Drops anti-correlated pairs entirely.
    Does NOT preserve PSD; safe for SymNMF (which only needs symmetric
    non-negative input) but not for methods requiring PSD input.
    """
    return np.maximum(C, 0.0)


def gaussian_kernel(C, sigma=None):
    """Gaussian similarity: K_ij = exp(-(1-C_ij)² / 2σ²).
    Perfectly correlated → 1, uncorrelated → exp(-1/2σ²).
    PSD by Mercer's theorem.
    """
    D = (1.0 - C) ** 2
    if sigma is None:
        sigma = np.median(np.sqrt(D[np.triu_indices_from(D, k=1)]))
        sigma = max(sigma, 1e-8)
    return np.exp(-D / (2 * sigma ** 2))


_TRANSFORMS = {
    "absolute":      absolute,
    "squared":       squared,
    "positive_part": positive_part,
    "gaussian":      gaussian_kernel,
}


# ── chunked correlation for large p ──────────────────────────────────

def _generate_X_block(A_block, Z, rng, n_rows, n_sample):
    """Generate one row-block of X = A @ Z + eps, centred."""
    eps = rng.standard_normal((n_rows, n_sample)).astype(np.float32)
    Xb = A_block @ Z + eps
    del eps
    mean = Xb.mean(axis=1, keepdims=True)
    Xb -= mean
    return Xb, mean.ravel()


def _regen_block(block_state, A_block, Z, n_sample):
    """Regenerate a centred X block from a cached RNG state."""
    rng = np.random.default_rng(0)
    rng.bit_generator.state = block_state
    return _generate_X_block(A_block, Z, rng, A_block.shape[0], n_sample)[0]


def _corr_tile_cpu(Xi, Xj, stds_i, stds_j, n):
    """Compute |corr| tile entirely on CPU."""
    cov = (Xi @ Xj.T).astype(np.float64) / (n - 1)
    return np.abs(cov / np.outer(stds_i, stds_j)).astype(np.float32)


def _generate_X_block_gpu(A_block, Z_gpu, block_seed, device):
    """Generate one centred X row-block entirely on GPU.

    Uses a per-block torch.Generator seeded deterministically so that
    the same block_seed always produces the same eps regardless of which
    physical GPU the block lands on.
    """
    import torch
    n_rows = A_block.shape[0]
    A_gpu = torch.from_numpy(np.ascontiguousarray(A_block)).to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(block_seed)
    eps = torch.randn(n_rows, Z_gpu.shape[1], device=device,
                      dtype=torch.float32, generator=gen)
    Xb = torch.mm(A_gpu, Z_gpu) + eps
    del eps, A_gpu
    Xb -= Xb.mean(dim=1, keepdim=True)
    return Xb


def _parse_gpu_ids(device):
    """Parse device string → list of GPU ids.

    "cpu"     → []
    "cuda"    → [0, 1, …, n-1]   (all available GPUs)
    "cuda:2"  → [2]
    """
    if device == "cpu":
        return []
    import torch
    if device == "cuda":
        return list(range(torch.cuda.device_count()))
    if device.startswith("cuda:"):
        return [int(device.split(":")[1])]
    raise ValueError(f"Unknown device: {device!r}")


# ── Pass 2 implementations ───────────────────────────────────────────

def _pass2_cpu(blocks, block_states, A, Z, n_sample, n, stds, Sigma):
    """Pass 2: all tiles on CPU."""
    for ib, (i0, i1) in enumerate(blocks):
        Xi = _regen_block(block_states[ib], A[i0:i1], Z, n_sample)
        for jb in range(ib, len(blocks)):
            j0, j1 = blocks[jb]
            Xj = _regen_block(block_states[jb], A[j0:j1], Z, n_sample)
            tile = _corr_tile_cpu(Xi, Xj, stds[i0:i1], stds[j0:j1], n)
            Sigma[i0:i1, j0:j1] = tile
            if ib != jb:
                Sigma[j0:j1, i0:i1] = tile.T
            del Xj, tile
        del Xi


def _pass1_gpu(blocks, A, Z, n_sample, block_seeds, device):
    """Pass 1 on GPU: compute per-row standard deviations."""
    import torch
    p = A.shape[0]
    stds = np.empty(p, dtype=np.float64)
    Z_gpu = torch.from_numpy(Z).to(device)

    for bid, (r0, r1) in enumerate(blocks):
        Xb = _generate_X_block_gpu(A[r0:r1], Z_gpu, block_seeds[bid], device)
        ss = torch.sum(Xb.double() ** 2, dim=1) / (n_sample - 1)
        stds[r0:r1] = torch.sqrt(ss).cpu().numpy()
        del Xb

    del Z_gpu
    stds[stds == 0] = 1.0
    return stds


def _gpu_worker_ib(ib, j_indices, blocks, block_seeds, A, Z_gpu, stds, n,
                   device):
    """Process all j-tiles for one i-block on one GPU.

    Xi is generated once and kept on GPU for all j-tiles in the batch.
    """
    import torch
    i0, i1 = blocks[ib]
    Xi = _generate_X_block_gpu(A[i0:i1], Z_gpu, block_seeds[ib], device)
    stds_i = torch.from_numpy(stds[i0:i1].astype(np.float64)).to(device)

    results = []
    for jb in j_indices:
        j0, j1 = blocks[jb]
        Xj = _generate_X_block_gpu(A[j0:j1], Z_gpu, block_seeds[jb], device)
        cov = torch.mm(Xi, Xj.T).double() / (n - 1)
        del Xj
        stds_j = torch.from_numpy(stds[j0:j1].astype(np.float64)).to(device)
        tile = torch.abs(cov / torch.outer(stds_i, stds_j)).float()
        results.append((jb, tile.cpu().numpy()))
        del cov, tile

    del Xi
    return results


def _pass2_gpu(blocks, block_seeds, A, Z, n_sample, n, stds, Sigma,
               gpu_ids):
    """Pass 2 on GPU(s): generate X blocks and compute tiles on-device.

    For each i-block, j-tiles are split across GPUs.  Each GPU
    generates Xi once, then processes its assigned j-tiles (generating
    Xj, matmul, normalise) entirely on-device.  Only the result tile
    is transferred back to CPU.
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_gpus = len(gpu_ids)
    Z_per_gpu = {g: torch.from_numpy(Z).to(f"cuda:{g}") for g in gpu_ids}

    with ThreadPoolExecutor(max_workers=max(n_gpus, 1)) as pool:
        for ib, (i0, i1) in enumerate(blocks):
            j_all = list(range(ib, len(blocks)))

            j_per_gpu = {g: [] for g in gpu_ids}
            for idx, jb in enumerate(j_all):
                j_per_gpu[gpu_ids[idx % n_gpus]].append(jb)

            futures = {}
            for gpu_id, j_indices in j_per_gpu.items():
                if not j_indices:
                    continue
                dev = f"cuda:{gpu_id}"
                fut = pool.submit(
                    _gpu_worker_ib, ib, j_indices, blocks, block_seeds,
                    A, Z_per_gpu[gpu_id], stds, n, dev,
                )
                futures[fut] = None

            for fut in as_completed(futures):
                for jb, tile in fut.result():
                    j0, j1 = blocks[jb]
                    Sigma[i0:i1, j0:j1] = tile
                    if ib != jb:
                        Sigma[j0:j1, i0:i1] = tile.T

    for Z_gpu in Z_per_gpu.values():
        del Z_gpu


# ── main chunked driver ──────────────────────────────────────────────

def _chunked_correlation(rng_state_before_eps, A, Z, p, n_sample, Sigma,
                         chunk_rows=_CHUNK_ROWS, device="cpu"):
    """Compute sample correlation tile-by-tile.

    Strategy
    --------
    CPU path  — Pass 1 caches numpy RNG states per block.  Pass 2
                regenerates Xi/Xj from cached states (all numpy).
    GPU path  — Per-block seeds derived via ``SeedSequence``.  Pass 1
                and Pass 2 generate X blocks on-device with
                ``torch.randn`` (cuRAND).  Only result tiles transfer
                back to CPU.

    Device modes
    ------------
    ``"cpu"``     — NumPy only.
    ``"cuda:0"``  — single GPU, full on-device pipeline.
    ``"cuda"``    — distribute tiles across **all** available GPUs.
    """
    n = n_sample
    blocks = [(r0, min(r0 + chunk_rows, p))
              for r0 in range(0, p, chunk_rows)]

    gpu_ids = _parse_gpu_ids(device)
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        import torch
        from numpy.random import SeedSequence

        # Derive per-block seeds from the RNG state at the eps-generation
        # point.  This doesn't consume the master rng (uses a copy).
        eps_rng = np.random.default_rng(0)
        eps_rng.bit_generator.state = rng_state_before_eps
        seed_base = int(eps_rng.integers(0, 2**63))
        block_seeds = [int(s.generate_state(1)[0])
                       for s in SeedSequence(seed_base).spawn(len(blocks))]

        primary_dev = torch.device(f"cuda:{gpu_ids[0]}")
        stds = _pass1_gpu(blocks, A, Z, n_sample, block_seeds, primary_dev)
        _pass2_gpu(blocks, block_seeds, A, Z, n_sample, n,
                   stds, Sigma, gpu_ids)
    else:
        # CPU path: cache numpy block states, pure numpy
        stds = np.empty(p, dtype=np.float64)
        block_states = []
        rng1 = np.random.default_rng(0)
        rng1.bit_generator.state = rng_state_before_eps

        for r0, r1 in blocks:
            block_states.append(rng1.bit_generator.state)
            Xb, _ = _generate_X_block(A[r0:r1], Z, rng1, r1 - r0, n_sample)
            stds[r0:r1] = np.sqrt(
                np.sum(Xb.astype(np.float64) ** 2, axis=1) / (n - 1))
            del Xb

        stds[stds == 0] = 1.0
        _pass2_cpu(blocks, block_states, A, Z, n_sample, n, stds, Sigma)

    for i in range(p):
        Sigma[i, i] = 1.0


# ── public API ────────────────────────────────────────────────────────

def corr_construct(X=None, p=None, n_sample=40000,
                   shrinkage=False, transform="absolute",
                   permute=True, random_state=None,
                   output_path=None, chunk_rows=_CHUNK_ROWS,
                   device="cpu"):
    """Construct a symmetric non-negative correlation-based matrix.

    Either pass real data X (p, n_sample) or set p to generate
    synthetic data via the same mixing-matrix model as TPDM:
    X = A @ Z, where A is a block-structured mixing matrix and
    Z has iid standard Gaussian entries.

    When ``permute=True`` (the default), the rows of the mixing matrix
    A are shuffled *before* computing Sigma, so the output is written
    directly in permuted order — no expensive post-hoc permutation of
    the p×p matrix is needed.

    Parameters
    ----------
    X : ndarray (p, n_sample), optional
        Time-series data.  Each row is one variable.  Permutation is
        not applied when real data is provided (the user is responsible
        for any desired reordering).
    p : int, optional
        Dimension (used to generate synthetic data when X is None).
    n_sample : int
        Number of observations (only used when X is None).
        Default 40,000 matches tpdm_construct (10 years of half-hourly
        snapshots: 250 days/yr × 8 h/day × 2/h).
    shrinkage : bool
        Apply Ledoit-Wolf shrinkage before converting to correlation.
        Recommended when p/n_sample > 0.1.  Not supported in chunked
        mode (p > chunk_rows); raises ValueError.
    transform : {"absolute", "squared", "positive_part", "gaussian"}
        Non-negative transformation.  In chunked mode only "absolute"
        is supported (applied during tile computation to avoid
        materialising the full p × p correlation matrix).
    permute : bool
        If True (default), randomly permute A's rows before computing
        Sigma — the output matrix is already in permuted order with
        block structure hidden.  Set to False for a block-ordered
        output (debug / visual inspection).  Ignored when X is
        provided.
    random_state : int or None
        Seed for full reproducibility (controls all internal RNGs).
    output_path : str or None
        If set, write the p×p result as a numpy memmap file at this path
        instead of returning an in-memory array.  Required for p >= ~100k.
    chunk_rows : int
        Row-block size for chunked generation (default 10,000).
    device : str
        ``"cpu"`` (default), ``"cuda:0"`` (single GPU), or ``"cuda"``
        (distribute tiles across **all** available GPUs).  GPU modes
        offload the heavy tile matmuls in chunked mode to PyTorch.

    Returns
    -------
    Sigma : ndarray or memmap (p, p), float32
        Symmetric non-negative matrix in [0, 1] with unit diagonal.
        When ``permute=True``, the matrix is already in permuted
        order — solvers can read it directly with no index indirection.
    perm : ndarray or None
        Permutation vector: ``perm[i]`` is the block-ordered index
        of the instrument at permuted position ``i``.  None when
        ``permute=False`` or when real data X was provided.
    block_labels : ndarray (p,), int32, or None
        Ground-truth cluster label for each row.  When
        ``permute=True``, labels are in permuted order (matching
        Sigma's row ordering).  None when real data X was provided.
    """
    rng = np.random.default_rng(random_state)

    use_chunked = (X is None and p is not None and p > chunk_rows)

    if use_chunked:
        if shrinkage:
            raise ValueError("Shrinkage is not supported in chunked mode "
                             "(p > chunk_rows).  Use shrinkage=False.")
        if transform != "absolute":
            raise ValueError(f"Chunked mode only supports transform='absolute', "
                             f"got '{transform}'.")

        k = int(np.floor(np.sqrt(p)))
        A, block_labels = _generate_block_mixing_matrix(rng, p, k)

        perm = None
        if permute:
            perm = rng.permutation(p)
            A = A[perm]
            block_labels = block_labels[perm]

        Z = rng.standard_normal((k + 1, n_sample)).astype(np.float32)

        rng_state_before_eps = rng.bit_generator.state

        Sigma = _allocate_output(p, output_path)
        _chunked_correlation(rng_state_before_eps, A, Z, p, n_sample,
                             Sigma, chunk_rows, device=device)

        del A, Z

        if output_path is not None:
            Sigma.flush()
            _save_companion_files(output_path, p, perm, block_labels)

        return Sigma, perm, block_labels

    # Small-p path
    perm = None
    block_labels = None
    if X is None:
        if p is None:
            raise ValueError("Provide either X or p.")
        k = int(np.floor(np.sqrt(p)))
        A, block_labels = _generate_block_mixing_matrix(rng, p, k)

        if permute:
            perm = rng.permutation(p)
            A = A[perm]
            block_labels = block_labels[perm]

        Z = rng.standard_normal((k + 1, n_sample)).astype(np.float32)
        eps = rng.standard_normal((p, n_sample)).astype(np.float32)
        X = A @ Z + eps

    p = X.shape[0]

    if shrinkage:
        Sigma, _ = shrunk_correlation(X)
    else:
        Sigma = sample_correlation(X)

    Sigma = (Sigma + Sigma.T) / 2

    if transform not in _TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform}. "
                         f"Choose from {list(_TRANSFORMS)}")
    Sigma = _TRANSFORMS[transform](Sigma).astype(np.float32)

    if output_path is not None:
        fp = np.memmap(output_path, dtype=np.float32, mode="w+",
                       shape=(p, p))
        fp[:] = Sigma
        fp.flush()
        Sigma = fp
        _save_companion_files(output_path, p, perm, block_labels)

    return Sigma, perm, block_labels


# ── demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = 6

    print("=== |correlation|, permuted (default) ===")
    Sigma, perm, labels = corr_construct(p=p, transform="absolute",
                                          permute=True, random_state=42)
    print(f"Diagonal (should be 1): {np.diag(Sigma)}")
    print(f"Range: [{Sigma.min():.4f}, {Sigma.max():.4f}]")
    print(f"Permutation (permuted→block-ordered): {perm}")
    print(f"Cluster labels: {labels}")
    print(Sigma, "\n")

    print("=== |correlation|, block-ordered (debug) ===")
    Sigma_bo, _, labels_bo = corr_construct(p=p, transform="absolute",
                                             permute=False, random_state=42)
    print(f"Range: [{Sigma_bo.min():.4f}, {Sigma_bo.max():.4f}]")
    print(f"Cluster labels (block-ordered): {labels_bo}")
    print(Sigma_bo, "\n")

    print("=== Gaussian kernel (no permutation) ===")
    Sigma_g, _, _ = corr_construct(p=p, transform="gaussian",
                                    permute=False, random_state=42)
    print(f"Diagonal: {np.diag(Sigma_g)}")
    print(f"Range: [{Sigma_g.min():.4f}, {Sigma_g.max():.4f}]")
    print(Sigma_g)
