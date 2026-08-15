# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline returns-matrix generator for ``clustering_through_time.ipynb``.

Stands in for an upstream ingest pipeline. It fabricates one heavy-tailed return
stream with *known* ground truth -- contiguous sector blocks, a mid-stream
reclassification, and a tail-stress (co-crash) episode -- and writes it to disk
as **parquet**, the format the notebook reads back on-GPU with cuDF. The planted
events have no analog in real data; they exist only so the clustering can be
checked against a known answer.

In production this file is replaced by your own ingest job and the notebook only
ever reads the returns matrix. Use ``scripts/make_returns_matrix.py`` to
(re)create the fixture from a repository checkout.

The notebook calls :func:`generate_returns_matrix` with its own config so the
on-disk returns matrix always matches the analysis parameters in the notebook.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

# Default fixture parameters. Keep these in sync with the notebook's config cell;
# the notebook passes its own values to generate_returns_matrix(), so these are only the
# defaults used when this script is run standalone.
SEED = 7
N = 50_000                  # instruments
K = 24                      # planted ground-truth #sectors (GICS-industry-group granularity)
L = 11_960                  # stream length (observations)
BREAK_T = 6_000             # a sector reclassification happens mid-stream
CRISIS = (9_000, 9_600)     # a tail-stress episode spans these observations
STRESS_SECTORS = (0, 1, 2, 3)   # sectors that quietly co-crash in the tail

DATA_DIR = "data"
RETURNS_PARQUET = "returns.parquet"
GT_PARQUET = "ground_truth.parquet"


def _pareto(shape, alpha, gen, device):
    """Heavy-tailed positive draws, matching ``np.random.pareto(a) + 1 == (1 - U)**(-1/a)``."""
    u = torch.rand(shape, device=device, generator=gen)
    return (1.0 - u).pow(-1.0 / alpha)


def _signed_pareto(shape, alpha, gen, device):
    """Signed heavy-tailed draws: Pareto magnitude x Rademacher sign.

    Real returns are signed and roughly symmetric, so factors and noise are
    built from signed heavy-tailed draws rather than strictly-positive Pareto.
    Positivity (required by the TPDM's angular framework) is restored *inside*
    the TPDM estimator via a marginal transform, not baked into the raw returns matrix.
    """
    mag = _pareto(shape, alpha, gen, device)
    sign = torch.where(torch.rand(shape, device=device, generator=gen) < 0.5,
                       -1.0, 1.0)
    return mag * sign


def simulate_returns(n, k, L, *, break_t, crisis, stress_sectors, device,
                     own_alpha=1.4, common=0.10, beta_lo=0.5, beta_hi=1.5,
                     fuzzy_frac=0.10, sec2_lo=0.3, sec2_hi=0.7,
                     p_crash=0.05, g_scale=6.0, reclass_frac=0.30,
                     noise_scale=0.2, clip_q=0.999, sector_scale_sigma=0.0,
                     seed=SEED):
    """One heavy-tailed *signed* return stream ``R`` (n x L) with contiguous sector
    blocks, a mid-stream reclassification, and a tail-stress episode. Returns ``R``
    and the sector labels (after reclassification).

    Each instrument's return is a sum of signed heavy-tailed factors:

    * a **broad market factor** (Gaussian, loading ``common`` + jitter), shared by all --
      deliberately *thin-tailed*, so it sets a realistic leading correlation eigenvalue
      without injecting any joint tail co-movement (the co-crash term alone owns the tail);
    * its **sector factor** scaled by a **per-instrument loading** ``beta_i ~
      U[beta_lo, beta_hi]`` -- so members of a sector co-move but are *not*
      identical copies, which is what makes the block structure (and hence the
      clustering problem) non-trivial. With ``sector_scale_sigma > 0`` the sector
      factors also get **heterogeneous per-sector strengths** (lognormal), so a
      few sectors dominate and the effective rank drops toward a realistic
      handful of factors rather than a flat ``k``-plateau;
    * for a ``fuzzy_frac`` minority, a **persistent secondary-sector exposure**,
      creating genuinely ambiguous ("between two groups") instruments;
    * a **tail co-crash** term: a shared *negative* shock to stressed sectors
      during the crisis window only;
    * **signed idiosyncratic noise**.

    Returns are generated signed, then **clipped after assembly** (winsorized at
    the ``clip_q`` quantile of |R|) -- bounding only the most extreme realized
    entries while preserving the joint tail structure the TPDM reads. Positivity
    is the TPDM estimator's responsibility (marginal transform), not the returns matrix's.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    sizes = torch.full((k,), n // k); sizes[: n - (n // k) * k] += 1
    sector = torch.repeat_interleave(torch.arange(k, device=device), sizes.to(device))

    # Post-break membership: a fraction of names reclassify into another sector.
    post = sector.clone()
    movers = torch.randperm(n, device=device, generator=gen)[: int(reclass_frac * n)]
    post[movers] = (sector[movers] + torch.randint(1, k, (movers.numel(),),
                                                    device=device, generator=gen)) % k

    # Market factor is Gaussian (thin-tailed): a shared mode that sets the leading
    # correlation eigenvalue but carries no joint tail dependence. Sector factors stay
    # signed heavy-tailed, so within-sector co-movement is return-like in body and tail.
    f_mkt = torch.randn((L,), device=device, generator=gen)  # broad market factor (Gaussian)
    f_sec = _signed_pareto((k, L), own_alpha, gen, device)   # per-sector factors (heavy-tailed)

    # Heterogeneous per-sector factor strengths. Real cross-sections are NOT
    # equal-weight: a handful of sectors dominate systematic variance while many
    # are minor. A lognormal per-sector scale (sigma>0) reproduces this, giving a
    # spectrum with a few large eigenvalues and a low effective rank (as in
    # Ghita et al., arXiv:2607.24518) rather than a flat k-plateau.
    # sigma=0 => equal weights.
    if sector_scale_sigma > 0:
        sec_scale = torch.exp(sector_scale_sigma
                              * torch.randn(k, device=device, generator=gen))
        f_sec = f_sec * sec_scale[:, None]

    # Per-instrument loadings: modest market exposure + heterogeneous sector beta.
    mkt_load = common + 0.10 * torch.rand(n, device=device, generator=gen)
    beta = beta_lo + (beta_hi - beta_lo) * torch.rand(n, device=device, generator=gen)

    def assemble(mem):
        return mkt_load[:, None] * f_mkt[None, :] + beta[:, None] * f_sec[mem]

    R = assemble(sector)                                    # pre-break membership
    R[:, break_t:] = assemble(post)[:, break_t:]           # post-break membership

    # Fuzzy membership: a minority carry a persistent secondary-sector exposure,
    # so they sit *between* two groups -- the marginal instruments a soft
    # factorization keeps distinct but a hard partition must commit.
    n_fuzzy = int(fuzzy_frac * n)
    if n_fuzzy > 0:
        fuzzy_idx = torch.randperm(n, device=device, generator=gen)[:n_fuzzy]
        secondary = (sector[fuzzy_idx] + torch.randint(1, k, (n_fuzzy,),
                                                       device=device, generator=gen)) % k
        beta2 = sec2_lo + (sec2_hi - sec2_lo) * torch.rand(n_fuzzy, device=device, generator=gen)
        R[fuzzy_idx] += beta2[:, None] * f_sec[secondary]

    # Tail co-crash: a shared NEGATIVE shock (drawdown) to stressed sectors during
    # the crisis only. Same direction across the block -> joint loss-tail dependence
    # the TPDM sees but the body-of-distribution correlation does not.
    ss = torch.tensor(list(stress_sectors), device=device)
    in_stress = (torch.isin(post, ss) if len(stress_sectors)
                 else torch.zeros(n, dtype=torch.bool, device=device))
    c0, c1 = crisis
    crash = torch.zeros(L, device=device)
    if c1 > c0:
        hit = torch.rand(c1 - c0, device=device, generator=gen) < p_crash
        crash[c0:c1][hit] = -g_scale * _pareto((int(hit.sum()),), 2.0, gen, device)
    R = R + in_stress[:, None].float() * crash[None, :]

    R = R + noise_scale * _signed_pareto((n, L), 3.0, gen, device)  # idiosyncratic noise

    # Generate-then-clip: winsorize only the most extreme realized entries. The
    # TPDM tail (top ~10% of observations by joint norm) sits well inside this
    # bound, so the planted co-crash structure is preserved.
    flat = R.abs().reshape(-1)
    m = flat.numel()
    samp = flat[torch.randint(m, (min(m, 1_000_000),), device=device, generator=gen)]
    clip_val = torch.quantile(samp, clip_q)
    R = R.clamp(-clip_val, clip_val)

    return R, post


def save_returns_matrix(R, sectors, data_dir=DATA_DIR):
    """Persist the returns matrix to parquet as an upstream ingest job would.
    Tries cuDF (GPU) -> parquet, falling back to pandas -> parquet. Parquet is
    chosen because cuDF reads it directly on the GPU."""
    os.makedirs(data_dir, exist_ok=True)
    returns_path = os.path.join(data_dir, RETURNS_PARQUET)
    gt_path = os.path.join(data_dir, GT_PARQUET)
    try:
        import cudf
        import cupy as cp
        cp_R = cp.asarray(R.contiguous().float())           # torch -> cupy (CUDA array interface)
        cols = [f"t{j}" for j in range(cp_R.shape[1])]
        cudf.DataFrame(cp_R, columns=cols).to_parquet(returns_path)
        cudf.DataFrame({"sector": cp.asarray(sectors.int().contiguous())}).to_parquet(gt_path)
        return "cuDF (GPU) -> parquet"
    except Exception:
        import pandas as pd
        pd.DataFrame(R.cpu().numpy()).to_parquet(returns_path)
        pd.DataFrame({"sector": sectors.cpu().numpy()}).to_parquet(gt_path)
        return "pandas -> parquet"


def generate_returns_matrix(data_dir=DATA_DIR, *, n=N, k=K, L=L, break_t=BREAK_T, crisis=CRISIS,
                   stress_sectors=STRESS_SECTORS, seed=SEED, regenerate=True, device=None):
    """Generate the fixture and write it to ``data_dir`` (idempotent).

    If ``regenerate`` is False and a returns matrix already exists on disk, this
    is a no-op. Returns a short human-readable description of what happened."""
    returns_path = os.path.join(data_dir, RETURNS_PARQUET)
    if not regenerate and os.path.exists(returns_path):
        return f"reusing existing returns matrix at {returns_path}"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R, sectors = simulate_returns(n, k, L, break_t=break_t, crisis=crisis,
                                  stress_sectors=stress_sectors, device=device, seed=seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fmt = save_returns_matrix(R, sectors, data_dir=data_dir)
    return f"wrote returns matrix {tuple(R.shape)} to {data_dir}/ via {fmt}"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--L", type=int, default=L)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    msg = generate_returns_matrix(args.data_dir, n=args.n, k=args.k, L=args.L, seed=args.seed,
                         regenerate=True)
    print(msg)


if __name__ == "__main__":
    main()
