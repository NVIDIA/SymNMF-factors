# Repository map

## Matrix construction and I/O

- `adaptgrow/corr_construction.py` generates synthetic correlation-style dependence
  matrices.
- `adaptgrow/tpdm_construction.py` generates synthetic tail-pairwise dependence matrices.
- `scripts/generate_benchmarks.py` writes deterministic matrix files and companion
  metadata.
- `adaptgrow/benchmark_loader.py` and `adaptgrow/matrix_io.py` load raw float32 memmaps. A matrix
  file may have `.meta.json`, `.perm.npy`, and `.labels.npy` companions.

## Solvers

- `adaptgrow/__init__.py` is the standalone AdaGrad / Block-SVRG AdaptGrow interface.
- `adaptgrow/_core.py` holds its private numerical kernels and solve state.
- `adaptgrow/distributed.py` provides row-sharded PyTorch/NCCL support.

AdaptGrow factors a symmetric, non-negative matrix `A` into a non-negative
matrix `H` such that `A ≈ H H.T`. It is a numerical primitive: callers own
factor definitions and downstream domain interpretation.

## Running code

- `notebooks/clustering_through_time.ipynb` is the primary synthetic-data demonstration.
- `scripts/run_distributed.py` runs AdaptGrow on prebuilt row-sharded matrices.
- `scripts/run_adaptgrow_reference.py` records a deterministic reference run.
- `tests/` includes CPU validation plus opt-in GPU and distributed checks.

## Generated artifacts

`data/`, `benchmark_output/`, notebook execution copies, traces, plots, and
per-run result dumps are intentionally ignored. Use
`configs/benchmark-small.json` as a template for recording a reproducible
experiment configuration. See `docs/runtime.md` for the container and
distributed environment.
