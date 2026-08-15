# Scripts

Run these from the repository root after `python -m pip install -e ".[dev]"`.

- `generate_benchmarks.py` writes synthetic dependence matrices.
- `make_returns_matrix.py` writes the synthetic notebook fixture.
- `run_adaptgrow_reference.py` records or checks a reference configuration.
- `run_distributed.py` launches the NCCL runner for prebuilt row-sharded matrices.
- `check_gds.py` validates optional KvikIO/cuFile loading.
- `launch.slurm` is the parameterized Slurm wrapper for the distributed runner.
