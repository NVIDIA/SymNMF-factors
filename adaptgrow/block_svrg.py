# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stochastic Block-SVRG SymNMF solver (the ``phi < 1`` regime).

Grows the batch fraction with the saturated-crawl controller and hands off to
:class:`~adaptgrow.adagrad.SymNMF_AdaGrad` once ``phi`` reaches full batch.
"""

from ._core import _LR_AUTO_MSG, _final_report, _init_solve, _run_block
from .adagrad import SymNMF_AdaGrad


class SymNMF_BlockSVRG:
    """Standalone stochastic Block-SVRG (phi < 1) SymNMF solver with the
    saturated-crawl growth controller. When ``phi`` grows to ``DENSE_FRAC`` it
    **hands off to** :class:`SymNMF_AdaGrad` to finish (dense), passing ``(H, G)``
    so the accumulator is NOT reset -- resetting it would spike the first dense
    step (~lr per coord) and can collapse H.

    ``optimize(A, k, H0=None, G0=None) -> H``.
    """

    def __init__(self, lr, entry_frac=0.5, eps=1e-8, max_iter=2000, tol=1e-5,
                 grad_tol=1e-4, use_svrg=True, snapshot_interval=None,
                 window_checks=5, eps_stall=1e-4, pg_gate=10.0, grow_factor=2.0,
                 check_interval=10, verbose=True):
        if isinstance(lr, str):
            raise ValueError(_LR_AUTO_MSG)
        self.lr = float(lr)
        self.entry_frac = float(entry_frac)
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.grad_tol = grad_tol
        self.use_svrg = use_svrg
        self.snapshot_interval = snapshot_interval
        self.window_checks = max(1, window_checks)
        self.eps_stall = eps_stall
        self.pg_gate = pg_gate
        self.grow_factor = grow_factor
        self.check_interval = check_interval
        self.verbose = verbose
        self.converged_ = False
        self.n_iters_ = 0
        self.grow_log_ = []

    def optimize(self, A, k, H0=None, G0=None):
        st = _init_solve(A, k, H0, G0, self.grad_tol)
        st, grew = _run_block(
            A, st, lr=self.lr, eps=self.eps, entry_frac=self.entry_frac,
            use_svrg=self.use_svrg, snapshot_interval=self.snapshot_interval,
            max_iter=self.max_iter, tol=self.tol, grad_tol=self.grad_tol,
            window_checks=self.window_checks, eps_stall=self.eps_stall,
            pg_gate=self.pg_gate, grow_factor=self.grow_factor,
            check_interval=self.check_interval, verbose=self.verbose)
        if grew:
            # phi reached 1 -> finish as full-batch AdaGrad, reusing (H, G) via the
            # shared state (accumulator continuous; global iter index continues).
            ada = SymNMF_AdaGrad(
                lr=self.lr, eps=self.eps, max_iter=self.max_iter, tol=self.tol,
                grad_tol=self.grad_tol, check_interval=self.check_interval,
                verbose=self.verbose)
            best_H = ada.optimize(A, k, _state=st)
            self.converged_ = ada.converged_
            self.n_iters_ = ada.n_iters_
            self.grow_log_ = ada.grow_log_
            return best_H
        _final_report(st, self.verbose)
        self.converged_ = st.converged
        self.n_iters_ = st.n_iters
        self.grow_log_ = st.grow_log
        return st.best_H
