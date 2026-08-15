# Tests

Run the deterministic CPU suite:

```bash
pytest -q -m "not gpu and not distributed"
```

It covers the public AdaptGrow interface, spectral probing, dense/block solver
paths, the distributed-matrix protocol, matrix generation, and the reference
runner.

CUDA precision and two-GPU runner checks are opt-in:

```bash
pytest -q -m gpu
pytest -q -m distributed
```

The distributed marker requires at least two CUDA devices and starts
`torch.distributed.run`; it is not run by the default CI workflow.
