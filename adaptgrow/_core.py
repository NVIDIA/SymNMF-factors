# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private numerical kernels and solve state for :mod:`adaptgrow`."""

import math
from collections import deque
from statistics import median

import torch
try:                                   # torch.distributed is part of torch, so
    import torch.distributed as _dist  # this keeps the file torch-only (no repo
except Exception:                      # imports) yet collective-safe at scale.
    _dist = None


def _agree(flag, device):
    """Broadcast rank-0's boolean so every rank takes the same control-flow
    branch. The convergence / phi-growth decisions are derived from per-GPU
    reductions (``torch.sum`` / ``torch.norm``) that are NOT bit-identical
    device-to-device, so a threshold can flip on one rank -- one rank then
    breaks (or grows phi) while the others do not, desynchronizing the
    collective stream and deadlocking NCCL. Forcing all ranks onto rank-0's
    decision keeps them in lockstep. No-op outside torch.distributed."""
    if _dist is None or not (_dist.is_available() and _dist.is_initialized()):
        return flag
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    _dist.broadcast(t, src=0)
    return bool(t.item())

# --- convergence / control constants (selection note + repo convention) ------
MAX_E_CONVERGED = 0.1     # loss gate: normalized error must be below this
STAGNATION_WARMUP = 50    # iters before objective-stagnation logic can fire
DENSE_FRAC = 1.0          # phi at/above which the exact dense gradient is used
N_REF = 10_000            # reference n for the scaled KKT threshold
BLOCK_GRAD_CHUNK = 8192   # max I-/J-dim per tile (peak ~chunk^2 * 4 bytes)


# ════════════════════════════════════════════════════════════════════════
# SymNMF math (dense torch.Tensor A)
# ════════════════════════════════════════════════════════════════════════

def _sqnorm(A):
    """||A||_F^2. Duck-types the distributed interface: a row-sharded matrix
    supplies ``A.sqnorm()`` (local sum + all_reduce), else dense fallback."""
    if hasattr(A, 'sqnorm'):
        return A.sqnorm()
    return torch.sum(A * A)


def _symnmf_error_from_ah(H, AH, A_sqnorm):
    """Normalized fitting error E = ||A - H H^T||_F^2 / ||A||_F^2, via the
    trace identity and a previously computed ``A @ H``."""
    with torch.no_grad():
        HtH = H.T @ H
        sq = A_sqnorm - 2.0 * torch.sum(H * AH) + torch.sum(HtH * HtH)
        return (sq.clamp(min=0) / A_sqnorm).item()


def _symnmf_error(A, H, A_sqnorm, AH=None):
    """Normalized fitting error, optionally reusing ``AH = A @ H``."""
    if AH is None:
        AH = A @ H
    return _symnmf_error_from_ah(H, AH, A_sqnorm)


def _grad_from_ah(H, AH):
    """Gradient from a previously computed ``A @ H``."""
    HtH = H.T @ H
    return torch.addmm(AH, H, HtH, beta=-4.0, alpha=4.0)


def _grad(A, H, AH=None):
    """grad f(H) = 4 (H (H^T H) - A H), optionally reusing ``A @ H``."""
    if AH is None:
        AH = A @ H
    return _grad_from_ah(H, AH)


def _projected_grad_norm(grad_H, H, boundary_eps=1e-16):
    """||grad_proj f(H)|| for H >= 0: zero the positive (infeasible) gradient
    components at the boundary (H ~ 0), keep the interior gradient."""
    proj = grad_H.clone()
    at_boundary = (H <= boundary_eps)
    proj[at_boundary & (grad_H > 0)] = 0.0
    return torch.norm(proj).item()


def _block_sample(n, entry_frac, device):
    """Row indices I and column indices J (without replacement),
    ``|I| = |J| = ceil(sqrt(entry_frac) * n)`` -- the (I x J) sub-block covers
    ~``entry_frac * n^2`` residual entries. Returned sorted so the same sample
    can be reused (SVRG evaluates the gradient at both H and the snapshot)."""
    bs = max(1, min(n, int(math.ceil(math.sqrt(entry_frac) * n))))
    I = torch.randperm(n, device=device)[:bs].sort().values
    J = torch.randperm(n, device=device)[:bs].sort().values
    return I, J


def _block_grad(A, H, entry_frac, I=None, J=None):
    """Block-structured stochastic gradient via dense matmuls (cuBLAS).

    For sampled row set I and column set J, the contribution of the (I x J)
    residual tile ``R_IJ = H_I H_J^T - A_IJ`` is

        grad[I] += 2 R_IJ   H_J
        grad[J] += 2 R_IJ^T H_I

    Unbiased up to the scalar ``entry_frac`` (absorbed by AdaGrad's per-element
    accumulator). Tiled at ``BLOCK_GRAD_CHUNK`` to bound peak memory.

    Distributed: a row-sharded ``A`` supplies ``A.block_entry_grad(H, I, J)``
    (owner-computes the I-rows it holds, then an all_reduce over the affected
    rows I u J) -- identical full gradient on every rank. ``I``/``J`` must match
    across ranks, which holds when every rank shares the sampling RNG state.
    """
    n = H.shape[0]
    if hasattr(A, 'block_entry_grad'):
        if I is None or J is None:
            I, J = _block_sample(n, entry_frac, H.device)
        return A.block_entry_grad(H, I, J)

    k = H.shape[1]
    if I is None or J is None:
        I, J = _block_sample(n, entry_frac, A.device)

    H_I = H[I]
    H_J = H[J]
    grad = torch.zeros_like(H)
    ch = BLOCK_GRAD_CHUNK
    bs = I.numel()

    for i0 in range(0, bs, ch):
        I_c = I[i0:i0 + ch]
        H_Ic = H_I[i0:i0 + ch]
        A_rows = A.index_select(0, I_c)
        for j0 in range(0, bs, ch):
            J_c = J[j0:j0 + ch]
            H_Jc = H_J[j0:j0 + ch]
            A_IJ = A_rows.index_select(1, J_c)
            R = torch.addmm(A_IJ, H_Ic, H_Jc.T, beta=-1.0, alpha=1.0)
            grad.index_add_(0, I_c, R @ H_Jc, alpha=2.0)
            grad.index_add_(0, J_c, R.T @ H_Ic, alpha=2.0)
        del A_rows

    return grad


# ════════════════════════════════════════════════════════════════════════
# Convergence check (strict 3-criterion AND, matching the repo engine)
# ════════════════════════════════════════════════════════════════════════

def _check_convergence(H, grad, E_k, prev_E, init_pg_norm, init_E,
                       tol, grad_tol, iteration):
    """Return ``(converged, metrics)``. Converged iff ALL three hold:

      1. loss gate:        E_k < MAX_E_CONVERGED
      2. scaled KKT:       ||grad_proj|| / (n k) < grad_tol * max(1, N_REF/n)
      3. objective stall:  0 <= (prev_E - E_k) / init_E < tol  (after warmup)

    ``metrics`` always carries ``E_k``, ``step_E``, ``pg_per_elem`` (the signals
    the growth controller reads).
    """
    with torch.no_grad():
        n, k = H.shape
        pg_norm = _projected_grad_norm(grad, H)
        if init_E is not None and init_E > 0 and math.isfinite(prev_E):
            step_E = (prev_E - E_k) / max(init_E, 1e-12)
        else:
            step_E = float('inf')
        pg_per_elem = pg_norm / max(n * k, 1)

    eff_grad_tol = grad_tol * max(1.0, N_REF / n)
    loss_ok = E_k < MAX_E_CONVERGED
    kkt_ok = pg_per_elem < eff_grad_tol
    past_warmup = iteration is None or iteration >= STAGNATION_WARMUP
    stall_ok = past_warmup and (0 <= step_E < tol)

    converged = bool(loss_ok and kkt_ok and stall_ok)
    metrics = dict(E_k=E_k, step_E=step_E, pg_per_elem=pg_per_elem,
                   pg_norm=pg_norm, rel_pg=pg_norm / init_pg_norm)
    return converged, metrics


# ════════════════════════════════════════════════════════════════════════
# Auto-configuration: spectral probe (phi_0).  lr is NOT auto -- it is a
# required, caller-supplied argument (see the module docstring for why).
# ════════════════════════════════════════════════════════════════════════

def spectral_probe(A, k, oversample=10, niter=4):
    r"""Cheap top-spectrum probe (~one iteration) for auto-configuration.

    Returns ``(gamma_kp1, r_eff)`` for symmetric (near-PSD) ``A`` via a
    randomized truncated SVD:

      * ``gamma_kp1 = lambda_{k+1} / |lambda_{k+2}|`` -- signal/noise gap at the
        rank-``k`` boundary. Large (>> 1) => clean dominant subspace => few
        iterations (selection note §3, §7).
      * ``r_eff`` -- effective rank as a **participation ratio** over the probed
        top-q spectrum, ``(sum s)^2 / sum(s^2)``. Low (-> 1) => a single common
        factor dominates => a random sub-block is a faithful gradient estimate
        (§4); ~m when m factors are comparable. Magnitude-weighted and k-free, so
        it is more robust than the older ``trace/lambda_1`` proxy.

    Returns ``(None, None)`` when the probe cannot run -- ``A`` is not a plain
    dense tensor (e.g. a row-sharded ``DistributedMatrix`` or a sparse shard) or
    is too small to resolve ``lambda_{k+2}``. The caller treats ``None`` as
    "gap unknown" (so it does NOT assume a clean gap). A genuinely clean/exact
    low-rank ``A`` returns ``gamma_kp1 = inf`` (``lambda_{k+2} ~ 0``).
    """
    n = A.size(0)
    is_dense = isinstance(A, torch.Tensor) and not A.is_sparse
    if not is_dense or (k + 2) > n:
        return None, None

    q = min(n, k + oversample + 2)
    with torch.no_grad():
        _, S, _ = torch.svd_lowrank(A, q=q, niter=niter)
        s = S.tolist()

    lk1 = s[k] if len(s) > k else 0.0
    lk2 = s[k + 1] if len(s) > k + 1 else 0.0
    gamma_kp1 = (lk1 / abs(lk2)) if lk2 > 0 else float('inf')
    # Effective rank as a magnitude-weighted *participation ratio* over the probed
    # top-q spectrum: (sum s)^2 / sum(s^2). This is ~1 when one factor dominates and
    # ~m for m comparable factors, interpolating smoothly. It is k-free and uses the
    # whole probed spectrum (not just s[0]), so it is more robust than the older
    # trace/lambda_1 proxy, which is sensitive only to the single top eigenvalue.
    ssum = float(S.sum().item())
    ssq = float((S * S).sum().item())
    r_eff = (ssum * ssum / ssq) if ssq > 0 else float('nan')
    return gamma_kp1, r_eff


# ════════════════════════════════════════════════════════════════════════
# Shared solve state + per-iteration primitives (used by both dedicated solvers)
# ════════════════════════════════════════════════════════════════════════

def _auto_snap(cur_frac):
    """SVRG snapshot cadence for a batch fraction: ~1/phi, floored at 10."""
    return max(10, int(math.ceil(1.0 / max(cur_frac, 1e-9))))


def _ada_update(H, G, g, lr, eps):
    """One projected-AdaGrad step: G += g**2 ; H <- clamp(H - lr*g/(sqrt(G)+eps)).
    The single fusable elementwise tail (cuTile target)."""
    G += g ** 2
    H = (H - lr * g / (torch.sqrt(G) + eps)).clamp_(min=1e-16)
    return H, G


class _SolveState:
    """Mutable state for one solve, threaded across the Block-SVRG -> AdaGrad
    handoff so the two phases are ONE continuous run: a shared *global* iteration
    index (``iter``), the AdaGrad accumulator ``G`` (never reset at handoff), the
    best iterate, and the init references the convergence test needs."""
    __slots__ = ("H", "AH", "G", "best_H", "best_AH", "best_E", "prev_E",
                 "init_pgn", "init_E", "A_sqnorm", "eff_grad_tol", "iter",
                 "n_iters", "converged", "grow_log")

    def __init__(self, H, AH, G, best_H, best_AH, best_E, prev_E,
                 init_pgn, init_E, A_sqnorm, eff_grad_tol):
        self.H, self.AH, self.G = H, AH, G
        self.best_H, self.best_AH, self.best_E = best_H, best_AH, best_E
        self.prev_E = prev_E
        self.init_pgn, self.init_E = init_pgn, init_E
        self.A_sqnorm, self.eff_grad_tol = A_sqnorm, eff_grad_tol
        self.iter = 0        # next GLOBAL iteration index to run
        self.n_iters = 0     # iterations completed (i + 1 of the last run)
        self.converged = False
        self.grow_log = []


def _init_solve(A, k, H0, G0, grad_tol):
    """Fresh solve state. Warm-start H from ``H0`` if given; seed G from ``G0``
    (else zero). Computes the init references (one A@H collective -- all ranks)."""
    n = A.size(0)
    device, dtype = A.device, A.dtype
    if H0 is not None:
        H = H0.to(device=device, dtype=dtype).clone().clamp_(min=1e-16)
    else:
        H = torch.abs(torch.rand(n, k, device=device, dtype=dtype))
    G = (torch.zeros_like(H) if G0 is None
         else G0.to(device=device, dtype=dtype).clone())
    A_sqnorm = _sqnorm(A)
    AH = A @ H
    init_grad = _grad_from_ah(H, AH)
    init_pgn = max(_projected_grad_norm(init_grad, H), 1e-12)
    init_E = _symnmf_error_from_ah(H, AH, A_sqnorm)
    return _SolveState(
        H=H, AH=AH, G=G, best_H=H.clone(), best_AH=AH.clone(),
        best_E=init_E, prev_E=init_E, init_pgn=init_pgn, init_E=init_E,
        A_sqnorm=A_sqnorm, eff_grad_tol=grad_tol * max(1.0, N_REF / n))


def _final_report(st, verbose):
    """Report the final projected-gradient norm without another ``A @ H``."""
    pgn = _projected_grad_norm(
        _grad_from_ah(st.best_H, st.best_AH), st.best_H)
    if verbose:
        print(f"=== Final: E={st.best_E:.4e}, ||grad_proj||={pgn:.3e}, "
              f"grow_log={st.grow_log} ===")


def _run_dense(A, st, *, lr, eps, max_iter, tol, grad_tol, check_interval, verbose):
    """Full-batch AdaGrad loop (phi = 1): branch-free, no sampling. Runs from
    ``st.iter`` (0 fresh, or the handoff iteration) to ``max_iter`` on the GLOBAL
    index so the convergence-check schedule (``i % check_interval``) is continuous
    across a Block-SVRG handoff. This is the clean per-iteration block the cuTile
    fusion / CUDA-graph capture target. Mutates and returns ``st``."""
    device = A.device
    for i in range(st.iter, max_iter):
        st.n_iters = i + 1
        with torch.no_grad():
            g = _grad_from_ah(st.H, st.AH)
            st.H, st.G = _ada_update(st.H, st.G, g, lr, eps)
            st.AH = A @ st.H
        E = _symnmf_error_from_ah(st.H, st.AH, st.A_sqnorm)
        if E < st.best_E:
            st.best_E = E
            st.best_H = st.H.clone()
            st.best_AH = st.AH.clone()
        if i % check_interval == 0:
            check_grad = _grad_from_ah(st.H, st.AH)
            converged, m = _check_convergence(
                st.H, check_grad, E, st.prev_E, st.init_pgn, st.init_E,
                tol, grad_tol, i)
            st.prev_E = E
            converged = _agree(converged, device)   # rank-consistent break
            if verbose:
                print(f"Iter {i:4d}: E={E:.4e}, step_E={m['step_E']:.2e}, "
                      f"pg/nk={m['pg_per_elem']:.2e}, phi=1")
            if converged:
                st.converged = True
                if m['E_k'] <= st.best_E:
                    st.best_E = m['E_k']
                    st.best_H = st.H.clone()
                    st.best_AH = st.AH.clone()
                if verbose:
                    print(f"Converged at iter {i} (E={st.best_E:.3e}, "
                          f"pg/nk={m['pg_per_elem']:.2e})")
                st.iter = i + 1
                return st
        st.iter = i + 1
    return st


def _run_block(A, st, *, lr, eps, entry_frac, use_svrg, snapshot_interval,
               max_iter, tol, grad_tol, window_checks, eps_stall, pg_gate,
               grow_factor, check_interval, verbose):
    """Stochastic Block-SVRG loop (phi < 1) + saturated-crawl growth controller.
    Grows phi on stagnation; when phi reaches ``DENSE_FRAC`` it stops and returns
    ``(st, grew_to_dense=True)`` with ``st.iter`` = the next (dense) iteration --
    the caller hands ``(H, G)`` to :func:`_run_dense`. Mutates ``st``."""
    device = A.device
    n = A.size(0)
    cur_frac = entry_frac
    mu = torch.zeros_like(st.H) if use_svrg else None
    snap = snapshot_interval if snapshot_interval is not None else _auto_snap(cur_frac)
    stepE_hist = deque(maxlen=window_checks)
    for i in range(st.iter, max_iter):
        st.n_iters = i + 1
        with torch.no_grad():
            if use_svrg:
                if i % snap == 0:
                    mu = _grad_from_ah(st.H, st.AH)
                I, J = _block_sample(n, cur_frac, device)
                g_block = _block_grad(A, st.H, cur_frac, I, J)
                affected = torch.cat([I, J]).unique()
                g = mu.clone()
                g[affected] = g_block[affected]
            else:
                g = _block_grad(A, st.H, cur_frac)
            st.H, st.G = _ada_update(st.H, st.G, g, lr, eps)
            st.AH = A @ st.H
        E = _symnmf_error_from_ah(st.H, st.AH, st.A_sqnorm)
        if E < st.best_E:
            st.best_E = E
            st.best_H = st.H.clone()
            st.best_AH = st.AH.clone()
        if i % check_interval == 0:
            check_grad = _grad_from_ah(st.H, st.AH)
            converged, m = _check_convergence(
                st.H, check_grad, E, st.prev_E, st.init_pgn, st.init_E,
                tol, grad_tol, i)
            st.prev_E = E
            converged = _agree(converged, device)
            if verbose:
                print(f"Iter {i:4d}: E={E:.4e}, step_E={m['step_E']:.2e}, "
                      f"pg/nk={m['pg_per_elem']:.2e}, phi={cur_frac:.3g}")
            if converged:
                st.converged = True
                if m['E_k'] <= st.best_E:
                    st.best_E = m['E_k']
                    st.best_H = st.H.clone()
                    st.best_AH = st.AH.clone()
                if verbose:
                    print(f"Converged at iter {i} (E={st.best_E:.3e}, "
                          f"pg/nk={m['pg_per_elem']:.2e})")
                st.iter = i + 1
                return st, False
            if i >= STAGNATION_WARMUP:
                stepE_hist.append(m['step_E'])
                if len(stepE_hist) >= window_checks:
                    crawl = (median(stepE_hist) < eps_stall
                             and m['pg_per_elem'] > pg_gate * st.eff_grad_tol)
                    crawl = _agree(crawl, device)   # rank-consistent phi growth
                    if crawl:
                        cur_frac = min(cur_frac * grow_factor, 1.0)
                        st.grow_log.append((i, cur_frac))
                        stepE_hist.clear()
                        if cur_frac >= DENSE_FRAC:
                            st.iter = i + 1          # hand off to dense AdaGrad
                            return st, True
                        if use_svrg and snapshot_interval is None:
                            snap = _auto_snap(cur_frac)
        st.iter = i + 1
    return st, False


_LR_AUTO_MSG = (
    "lr must be a float; 'auto' is not supported. AdaGrad's usable-lr window is "
    "narrow and scale-dependent and a short probe cannot distinguish the "
    "dead-zone escape from a permanent collapse, so there is no reliable "
    "auto-bracket. Pass an explicit lr.")
