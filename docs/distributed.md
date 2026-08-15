# Distributed AdaptGrow

`run_distributed.py` applies the same `AdaptGrow` implementation to a
row-sharded symmetric matrix. Each rank owns rows of `S`, while the factor
matrix remains replicated across ranks. PyTorch distributed/NCCL performs the
required `all_gather` and `all_reduce` operations.

This path factorizes pre-built matrix files only. Distributed rank selection,
in-job streaming matrix construction, and distributed spherical K-means are
not implemented.

## Requirements

- Two or more CUDA devices for the smoke check.
- PyTorch with NCCL support and a compatible CUDA driver/runtime.
- Matrix dimensions divisible by the number of ranks.
- For multi-node runs, working NCCL network transport and a shared matrix
  directory.

See [`docs/runtime.md`](../docs/runtime.md) for the public NGC container setup
and runtime-recording requirements.

## Single-node smoke check

Generate a small matrix from the repository root:

```bash
python scripts/generate_benchmarks.py --sizes 100 --seeds 42 --matrix-types corr
```

Then launch one rank per GPU:

```bash
torchrun --standalone --nproc-per-node=2 \
  scripts/run_distributed.py \
  --matrix-dir data/benchmarks --kind corr --k 10 --n-iter 100 \
  --io-method cpu
```

The runner accepts raw float32 `.dat` memmaps with the companion metadata
written by `scripts/generate_benchmarks.py`. Use `--suffix` to select another file
suffix. Default `--io-method auto` tries GDS on CUDA and falls back to CPU;
use `--io-method cpu` for the smoke check above, and `--io-method gds` only on
a validated KvikIO/cuFile installation.

## Slurm

`scripts/launch.slurm` is a parameterized template. From a checkout installed in the
selected container/environment:

```bash
MATRIX_DIR=/shared/path/to/matrices sbatch scripts/launch.slurm
```

Set the allocation directives in `scripts/launch.slurm` to match the cluster. The
template supplies ranks through Slurm environment variables and invokes
`scripts/run_distributed.py`; it contains no private path, account, or
container assumptions.
