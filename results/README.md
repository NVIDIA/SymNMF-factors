# Reference benchmark summaries

Reference outputs are generated locally rather than committed. Create one from
the public implementation:

```bash
python scripts/run_adaptgrow_reference.py
```

This writes `reference/adaptgrow_summary.csv` and a runtime metadata sidecar.
To compare a new run with a prior result, use:

```bash
python scripts/run_adaptgrow_reference.py --check path/to/adaptgrow_summary.csv
```

The check compares reconstruction error only; wall-clock time is
hardware-dependent and is recorded for context, not compared.

For each published run, capture:

- the matrix kind, size, seed, and generation parameters;
- solver name, rank, and hyperparameters;
- precision and TF32 settings;
- PyTorch/CUDA versions and device details; and
- code revision and configuration file.

Per-run label arrays, convergence traces, plots, and full sweep outputs are
local artifacts and are excluded from version control.
