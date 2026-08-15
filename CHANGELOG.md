# Changelog

All notable changes to SymNMF-factors will be documented in this file.

The project follows [Semantic Versioning](https://semver.org/) after its first
public release.

## Unreleased

### Added

- Standalone AdaptGrow SymNMF implementation combining projected AdaGrad and
  Block-SVRG.
- Spectral probing and adaptive optimization-path selection.
- Synthetic correlation and tail-dependence matrix generators.
- PyTorch/NCCL row-sharded multi-GPU and multi-node factorization.
- Reproducible reference runner, tests, runtime documentation, and financial
  application notebook using synthetic data.

### Changed

- Reorganized the source into `adaptgrow/`, `scripts/`, `notebooks/`, `docs/`,
  and `tests/`.
- Reused `A @ H` products across gradient, objective, convergence, and SVRG
  calculations.
