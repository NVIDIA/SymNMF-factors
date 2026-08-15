# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-batch AdaGrad SymNMF solver (the ``phi = 1`` regime)."""

from ._core import _LR_AUTO_MSG, _final_report, _init_solve, _run_dense


class SymNMF_AdaGrad:
    """Standalone full-batch (phi = 1) projected-AdaGrad SymNMF solver.

    ``H <- clamp(H - lr*g/(sqrt(G)+eps), min=0)`` with ``g = 4(H HtH - AH)`` and
    the diagonal accumulator ``G``. Branch-free, no sampling -- the clean loop the
    cuTile fused kernel + CUDA-graph capture target. Also the hand-off target of
    :class:`SymNMF_BlockSVRG` at ``phi = 1`` (accepts a warm ``G0`` so the
    accumulator, hence the step scale, is continuous across the handoff).

    ``optimize(A, k, H0=None, G0=None) -> H``.
    """

    def __init__(self, lr, eps=1e-8, max_iter=2000, tol=1e-5, grad_tol=1e-4,
                 check_interval=10, verbose=True):
        if isinstance(lr, str):
            raise ValueError(_LR_AUTO_MSG)
        self.lr = float(lr)
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.grad_tol = grad_tol
        self.check_interval = check_interval
        self.verbose = verbose
        self.converged_ = False
        self.n_iters_ = 0
        self.grow_log_ = []

    def optimize(self, A, k, H0=None, G0=None, _state=None):
        # ``_state`` (internal): resume from a handed-off Block-SVRG solve so the
        # two phases form one continuous run. Public callers pass H0/G0.
        st = _state if _state is not None else _init_solve(A, k, H0, G0, self.grad_tol)
        st = _run_dense(A, st, lr=self.lr, eps=self.eps, max_iter=self.max_iter,
                        tol=self.tol, grad_tol=self.grad_tol,
                        check_interval=self.check_interval, verbose=self.verbose)
        _final_report(st, self.verbose)
        self.converged_ = st.converged
        self.n_iters_ = st.n_iters
        self.grow_log_ = st.grow_log
        return st.best_H
