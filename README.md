# SymNMF-factors

GPU-accelerated symmetric non-negative matrix factorization for discovering
latent factors in large similarity and dependence matrices. The AdaptGrow
solver and the synthetic correlation / TPDM generators follow
[Ghita et al. (2026)](https://arxiv.org/abs/2607.24518).

## Overview

SymNMF-factors provides three primary components:

1. A standalone AdaptGrow solver for single-GPU execution. AdaptGrow combines
   projected AdaGrad with block-sampled stochastic variance-reduced gradient
   (Block-SVRG), uses a spectral probe to select the initial optimization path,
   and can transition toward full-batch AdaGrad.
2. PyTorch/NCCL support for running AdaptGrow across multiple GPUs and nodes on
   prebuilt, row-sharded matrices.
3. A notebook demonstrating a financial application by extracting and tracking
   latent factors from synthetic correlation and tail-dependence matrices.

The project supplies numerical primitives only. It does not define financial
factors, neutralization, shrinkage, portfolio rules, trading strategies, or
financial advice.

## Getting Started

Clone the repository and install it into a Python environment:

```bash
git clone https://github.com/NVIDIA/SymNMF-factors.git
cd SymNMF-factors
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python - <<'PY'
import torch
from adaptgrow import AdaptGrow
from adaptgrow.corr_construction import corr_construct

matrix, _, _ = corr_construct(p=100, random_state=42)
matrix = torch.as_tensor(matrix, dtype=torch.float32)
factors = AdaptGrow(lr=2.0, max_iter=200).optimize(matrix, k=10)
print(factors.shape)
PY
```

For a container-based setup, use the public NGC PyTorch image documented in
[`docs/runtime.md`](docs/runtime.md).

## Requirements

- Linux is recommended for CUDA and distributed execution.
- Python 3.10 or later.
- PyTorch 2.2 or later, NumPy 1.24 or later, and SciPy 1.10 or later.
- A CUDA-capable NVIDIA GPU and compatible NVIDIA driver for practical
  workloads.
- NCCL and one process per GPU for distributed execution.

Optional extras:

- `.[notebook]` installs Jupyter, plotting, parquet, and pandas support.
- `.[gds]` enables KvikIO/cuFile-backed matrix loading.
- `.[rapids]` enables optional GPU dataframe ingestion.

See [`docs/runtime.md`](docs/runtime.md) for container, CUDA, NCCL, and
reproducibility requirements.

## Usage

The public solver interface is:

```python
from adaptgrow import AdaptGrow

solver = AdaptGrow(lr=2.0, entry_frac="auto", max_iter=2_000)
factors = solver.optimize(matrix, k=rank)
print(solver.resolved_)
```

Additional workflows:

- Notebook: [`notebooks/clustering_through_time.ipynb`](notebooks/clustering_through_time.ipynb)
- Distributed execution: [`docs/distributed.md`](docs/distributed.md)
- Runtime and container setup: [`docs/runtime.md`](docs/runtime.md)
- Repository map: [`docs/repo_map.md`](docs/repo_map.md)
- Reproducible reference runner: `python scripts/run_adaptgrow_reference.py`

## Testing

Run deterministic CPU checks:

```bash
pytest -q -m "not gpu and not distributed"
```

CUDA and multi-GPU tests are opt-in:

```bash
pytest -q -m "gpu and not distributed"
pytest -q -m distributed
```

## Releases and Roadmap

Release changes are tracked in [`CHANGELOG.md`](CHANGELOG.md). Near-term work
focuses on validating the public release, expanding reproducible performance
results, and hardening distributed execution for supported configurations.

## Contribution Guidelines

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before contributing.

## Governance and Maintainers

The project follows maintainer-led governance described in
[`GOVERNANCE.md`](GOVERNANCE.md). See [`MAINTAINERS.md`](MAINTAINERS.md) for
project ownership.

## Security

Do not report vulnerabilities through public issues. Follow
[`SECURITY.md`](SECURITY.md) to contact NVIDIA Product Security.

## Support

SymNMF-factors is maintained with best-effort support through repository
issues. Supported scope and response expectations are documented in
[`SUPPORT.md`](SUPPORT.md).

## Community

Use repository issues for bug reports, feature requests, and technical
questions. Please follow the Code of Conduct in all project interactions.

## References

- Lavinia Ghita, Dhruv Desai, Jake Goldberg, and Roman Yokunda Enzmann.
  [Low-Rank Dependence Decomposition via Accelerated Symmetric Non-negative
  Matrix Factorization](https://arxiv.org/abs/2607.24518).
  arXiv:2607.24518, 2026.
- [PyTorch](https://pytorch.org/)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [NVIDIA NGC PyTorch container (`26.04-py3`)](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/pytorch/26.04-py3)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE)
for the full license text.
