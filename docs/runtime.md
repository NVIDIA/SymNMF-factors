# Runtime requirements

## Supported environment

AdaptGrow is developed against:

- Python 3.10 or later;
- PyTorch 2.2 or later;
- NumPy 1.24 or later and SciPy 1.10 or later; and
- an NVIDIA CUDA GPU for practical single-GPU and distributed workloads.

CPU execution is sufficient for the small pytest checks and examples. GPU
execution requires an NVIDIA driver compatible with the CUDA runtime in the
selected container. The host also needs the NVIDIA Container Toolkit when
using Docker.

## NGC PyTorch container

Use a public NGC PyTorch image rather than a private registry image. At the
time this guide was written, `nvcr.io/nvidia/pytorch:26.04-py3` is a suitable
starting point; select a tag compatible with the host driver and record the
immutable digest for experiments.

After authenticating to NGC, launch a checkout with:

```bash
export CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
docker run --gpus all --rm -it \
  -v "$PWD":/workspace -w /workspace \
  "$CONTAINER_IMAGE" bash
```

Inside the container:

```bash
python -m pip install -e ".[dev,notebook]"
pytest -q -m "not gpu and not distributed"
python scripts/run_adaptgrow_reference.py
```

KvikIO/GDS and RAPIDS are optional:

```bash
python -m pip install -e ".[gds]"
python -m pip install --extra-index-url https://pypi.nvidia.com ".[rapids]"
```

## Record each run

Store the following beside every benchmark or reference result:

```bash
git rev-parse HEAD
docker image inspect "$CONTAINER_IMAGE" --format '{{index .RepoDigests 0}}'
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("nccl:", torch.cuda.nccl.version() if torch.cuda.is_available() else None)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("tf32:", torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None)
PY
```

The reference runner writes the code revision, runtime, device, container
image environment variable, and TF32 setting to its metadata sidecar.

## Distributed requirements

The distributed path uses `torchrun`, NCCL, and one process per GPU.

- Every rank needs a CUDA device and the same compatible PyTorch/CUDA/NCCL
  runtime.
- The input matrix dimension must be divisible by the world size.
- Multi-node jobs need working NCCL network transport and a shared filesystem
  containing the input memmaps.
- GPU Direct Storage requires a compatible filesystem and KvikIO/cuFile; use
  `--io-method cpu` when that stack is unavailable.

Start with the two-GPU smoke command in
[`docs/distributed.md`](distributed.md). The multi-node workflow
requires a user-provided Slurm allocation and compatible NCCL networking.
