# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic CPU correctness checks for the standalone AdaptGrow solver."""

import math

import pytest
import torch

from adaptgrow import (
    AdaptGrow,
    SymNMF_AdaGrad,
    SymNMF_BlockSVRG,
    _block_grad,
    _sqnorm,
    _symnmf_error,
    spectral_probe,
)


SEED = 42


def _planted_matrix(n=200, k=8):
    """Return an exact-rank non-negative symmetric matrix on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    factors = torch.abs(torch.randn(n, k, generator=generator))
    return factors @ factors.T


class _DuckShard:
    """Single-process stand-in for the distributed matrix interface."""

    def __init__(self, matrix):
        self._matrix = matrix
        self.calls = {"sqnorm": 0, "block_entry_grad": 0, "matmul": 0}

    def size(self, dim=None):
        return self._matrix.size(dim) if dim is not None else self._matrix.size()

    @property
    def device(self):
        return self._matrix.device

    @property
    def dtype(self):
        return self._matrix.dtype

    @property
    def is_sparse(self):
        return False

    def __matmul__(self, factors):
        self.calls["matmul"] += 1
        return self._matrix @ factors

    def sqnorm(self):
        self.calls["sqnorm"] += 1
        return torch.sum(self._matrix * self._matrix)

    def block_entry_grad(self, factors, rows, columns):
        self.calls["block_entry_grad"] += 1
        return _block_grad(self._matrix, factors, None, rows, columns)


@pytest.fixture
def planted_matrix():
    torch.manual_seed(SEED)
    return _planted_matrix()


def test_spectral_probe_returns_finite_effective_rank(planted_matrix):
    gap, effective_rank = spectral_probe(planted_matrix, k=8)

    assert math.isfinite(gap)
    assert gap > 0
    assert effective_rank >= 1.0


def test_auto_configuration_uses_dense_adagrad_for_small_matrix(planted_matrix):
    torch.manual_seed(SEED)
    solver = AdaptGrow(
        lr=2.0,
        max_iter=3000,
        tol=1e-5,
        grad_tol=1e-4,
        verbose=False,
    )
    factors = solver.optimize(planted_matrix, k=8)
    error = _symnmf_error(planted_matrix, factors, _sqnorm(planted_matrix))

    assert solver.resolved_["entry_frac"] == 1.0
    assert solver.grow_log_ == []
    assert torch.all(factors >= 0)
    assert error < 1e-2


def test_block_svrg_grows_to_dense_adagrad(planted_matrix):
    torch.manual_seed(SEED)
    solver = AdaptGrow(
        lr=2.0,
        entry_frac=0.25,
        max_iter=4000,
        tol=1e-5,
        grad_tol=1e-4,
        verbose=False,
    )
    factors = solver.optimize(planted_matrix, k=8)
    error = _symnmf_error(planted_matrix, factors, _sqnorm(planted_matrix))

    assert torch.all(factors >= 0)
    assert solver.grow_log_
    assert solver.grow_log_[-1][1] == 1.0
    assert error < 5e-2


@pytest.mark.parametrize("solver_type", [SymNMF_AdaGrad, SymNMF_BlockSVRG])
def test_direct_solver_interfaces_return_nonnegative_factors(solver_type):
    matrix = _planted_matrix(n=32, k=3)
    torch.manual_seed(SEED)
    if solver_type is SymNMF_AdaGrad:
        solver = solver_type(lr=1.0, max_iter=5, check_interval=1, verbose=False)
    else:
        solver = solver_type(
            lr=1.0,
            entry_frac=1.0,
            max_iter=5,
            check_interval=1,
            verbose=False,
        )

    factors = solver.optimize(matrix, k=3)

    assert factors.shape == (32, 3)
    assert torch.all(factors >= 0)


def test_dense_solver_computes_one_full_product_per_iteration():
    matrix = _planted_matrix(n=32, k=3)
    counted_matrix = _DuckShard(matrix)
    torch.manual_seed(SEED)
    solver = SymNMF_AdaGrad(
        lr=1.0,
        max_iter=5,
        check_interval=1,
        verbose=False,
    )

    solver.optimize(counted_matrix, k=3)

    # One initial A @ H, then one product after each update. Error,
    # convergence, and final reporting must reuse those products.
    assert counted_matrix.calls["matmul"] == 6


def test_duck_typed_distributed_interface_uses_block_gradient(planted_matrix):
    torch.manual_seed(SEED)
    shard = _DuckShard(planted_matrix)
    solver = AdaptGrow(
        lr=2.0,
        entry_frac=0.25,
        max_iter=20,
        check_interval=1,
        verbose=False,
    )

    solver.optimize(shard, k=8)

    assert shard.calls["sqnorm"] >= 1
    assert shard.calls["block_entry_grad"] >= 1
    assert shard.calls["matmul"] == 21


def test_invalid_learning_rate_configuration_is_rejected():
    with pytest.raises(ValueError):
        AdaptGrow(lr="auto")
    with pytest.raises(TypeError):
        AdaptGrow()
