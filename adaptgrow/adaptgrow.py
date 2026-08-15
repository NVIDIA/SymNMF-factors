# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AdaptGrow orchestrator: probe ``phi_0``, dispatch to the dedicated solver."""

from ._core import DENSE_FRAC, spectral_probe
from .adagrad import SymNMF_AdaGrad
from .block_svrg import SymNMF_BlockSVRG


class AdaptGrow:
    r"""AdaptGrow: a unified, auto-configured AdaGrad <-> Block-SVRG solver.

    The name is "adaptive grow": an AdaGrad backbone whose **batch fraction**
    ``phi`` (the fraction of the residual it samples per step) adaptively grows
    from Block-SVRG (stochastic, ``phi < 1``) up to full-batch AdaGrad
    (``phi = 1``) -- the "AdaptGrow" controller of the selection note. AdaGrad is
    the ``phi = 1`` limit, so this single solver subsumes both parents.

    Distributed / multinode: matches the parent solvers by duck-typing the
    row-sharded interface (``A @ H``, ``A.sqnorm()``, ``A.block_entry_grad``),
    so passing a ``DistributedMatrix`` runs it multinode with no code change.
    Every rank must share the sampling RNG state (same seed) so the sampled
    blocks agree -- the same requirement the parent stochastic solvers carry.

    Parameters
    ----------
    lr : float (required)
        Step size. There is no ``"auto"``: AdaGrad's usable-lr window is narrow
        and scale-dependent, so the caller must pass a pinned lr (passing the
        string ``"auto"`` raises).
    entry_frac : float or "auto"
        Batch fraction ``phi_0`` -- the single knob that selects the regime:
          * ``"auto"`` -> the spectral probe seeds it (§7): ``1.0`` if
            ``n <= crossover_n`` or ``gamma_{k+1} >= gamma_threshold``, else ``0.5``.
          * ``1.0``    -> full-batch **AdaGrad** (the ``phi = 1`` limit).
          * ``< 1.0``  -> stochastic **Block-SVRG** (the ``phi < 1`` regime),
            with the controller free to grow ``phi`` toward 1.
    eps : float
        AdaGrad accumulator floor.
    max_iter, tol, grad_tol :
        Iteration cap and the stagnation / scaled-KKT tolerances of the
        3-criterion convergence test.
    crossover_n, gamma_threshold :
        Auto-``phi_0`` seeding thresholds (§7, §11.6).
    probe_metric, reff_threshold :
        Which spectral statistic drives the auto-``phi_0`` dense/stochastic switch:
        ``"gap"`` (default) uses ``gamma_{k+1} >= gamma_threshold``; ``"reff"`` uses
        the k-free participation-ratio effective rank ``r_eff >= reff_threshold``
        (dense/full-batch when many comparable factors, stochastic when one blob).
    use_svrg : bool
        SVRG control variate in the stochastic phase (default True).
    snapshot_interval : int or None
        SVRG full-gradient cadence; ``None`` auto-tunes to ``max(10, ceil(1/phi))``.
    window_checks, eps_stall, pg_gate, grow_factor :
        ``saturated_crawl`` controller knobs (§11.7): grow ``phi`` by
        ``grow_factor`` when the median ``step_E`` over ``window_checks`` checks
        is ``< eps_stall`` AND ``pg/(nk) > pg_gate * eff_grad_tol``.
    check_interval : int
        Iterations between convergence checks / controller updates.

    After :meth:`optimize`, ``resolved_`` holds the chosen ``lr`` / ``phi_0`` /
    probe, and ``grow_log_`` the ``(iter, phi)`` growth events.
    """

    def __init__(self, lr, entry_frac="auto", eps=1e-8,
                 max_iter=2000, tol=1e-5, grad_tol=1e-4,
                 crossover_n=100_000, gamma_threshold=5.0,
                 probe_metric="gap", reff_threshold=3.0,
                 use_svrg=True, snapshot_interval=None,
                 window_checks=5, eps_stall=1e-4, pg_gate=10.0,
                 grow_factor=2.0, check_interval=10, verbose=True):
        if isinstance(lr, str):
            raise ValueError(
                "lr must be a float; 'auto' is not supported. AdaGrad's usable-lr "
                "window is narrow and scale-dependent and a short probe cannot "
                "distinguish the dead-zone escape from a permanent collapse, so "
                "there is no reliable auto-bracket. Pass an explicit lr.")
        self.lr = float(lr)
        self.entry_frac = entry_frac
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.grad_tol = grad_tol
        self.crossover_n = crossover_n
        self.gamma_threshold = gamma_threshold
        self.probe_metric = probe_metric
        self.reff_threshold = reff_threshold
        self.use_svrg = use_svrg
        self.snapshot_interval = snapshot_interval
        self.window_checks = max(1, window_checks)
        self.eps_stall = eps_stall
        self.pg_gate = pg_gate
        self.grow_factor = grow_factor
        self.check_interval = check_interval
        self.verbose = verbose
        self.resolved_ = {}
        self.grow_log_ = []
        self.converged_ = False
        self.n_iters_ = 0

    def _seed_phi0(self, A, k):
        if self.entry_frac != "auto":
            return float(self.entry_frac), None
        n = A.size(0)
        gamma_kp1, r_eff = spectral_probe(A, k)
        # gamma_kp1 / r_eff are None when the probe could not run (distributed /
        # sparse / too small): treat the structure as unknown, i.e. do NOT assume a
        # clean/dense-friendly spectrum, so a large-n shard correctly starts
        # stochastic (phi_0 = 0.5).
        clean_gap = (gamma_kp1 is not None) and (gamma_kp1 >= self.gamma_threshold)
        # k-free effective-rank rule: a high participation ratio (many comparable
        # factors) => a random sub-block misses structure => full AdaGrad; a low one
        # (one dominant blob) => sub-block is faithful => Block-SVRG.
        reff_ok = (r_eff is not None) and (r_eff >= self.reff_threshold)
        dense_ok = reff_ok if self.probe_metric == "reff" else clean_gap
        small = n <= self.crossover_n
        phi0 = 1.0 if (small or dense_ok) else 0.5
        return phi0, dict(gamma_kp1=gamma_kp1, r_eff=r_eff, small=small,
                          clean_gap=clean_gap, reff_ok=reff_ok,
                          metric=self.probe_metric)

    def optimize(self, A, k, H0=None):
        # Probe seeds phi_0, then dispatch to the dedicated solver. AdaGrad is the
        # phi = 1 limit, so a dense seed routes to SymNMF_AdaGrad; a stochastic
        # seed routes to SymNMF_BlockSVRG, which self-hands-off to AdaGrad if it
        # grows phi to 1. The orchestrator holds no per-iteration state -- the
        # sub-solver owns the loop and reports converged_/n_iters_/grow_log_ back.
        phi0, probe = self._seed_phi0(A, k)
        self.resolved_ = dict(lr=self.lr, entry_frac=phi0, probe=probe)
        if self.verbose:
            msg = f"[AdaptGrow] phi_0={phi0:.3g}, lr={self.lr:.3g}"
            if probe is not None and probe['gamma_kp1'] is not None:
                msg += (f"  (gamma_kp1={probe['gamma_kp1']:.3g}, "
                        f"r_eff={probe['r_eff']:.3g})")
            elif probe is not None:
                msg += "  (spectral probe unavailable: seeded by scale)"
            print(msg)

        if phi0 >= DENSE_FRAC:
            solver = SymNMF_AdaGrad(
                lr=self.lr, eps=self.eps, max_iter=self.max_iter, tol=self.tol,
                grad_tol=self.grad_tol, check_interval=self.check_interval,
                verbose=self.verbose)
        else:
            solver = SymNMF_BlockSVRG(
                lr=self.lr, entry_frac=phi0, eps=self.eps, max_iter=self.max_iter,
                tol=self.tol, grad_tol=self.grad_tol, use_svrg=self.use_svrg,
                snapshot_interval=self.snapshot_interval,
                window_checks=self.window_checks, eps_stall=self.eps_stall,
                pg_gate=self.pg_gate, grow_factor=self.grow_factor,
                check_interval=self.check_interval, verbose=self.verbose)

        best_H = solver.optimize(A, k, H0)
        self.converged_ = solver.converged_
        self.n_iters_ = solver.n_iters_
        self.grow_log_ = solver.grow_log_
        return best_H


# Backwards-compatible alias: the solver was formerly referred to as ``AdaGrow``.
AdaGrow = AdaptGrow
