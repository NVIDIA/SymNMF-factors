# Support

## Support Level

SymNMF-factors is maintained by NVIDIA. Support is provided on a best-effort
basis through the repository issue tracker; no response-time or
resolution-time service-level agreement is provided.

## Supported Scope

Supported project functionality includes:

- Single-GPU AdaptGrow factorization.
- Synthetic matrix generation and matrix loading.
- The documented notebook and reference-runner workflows.
- Multi-GPU and multi-node AdaptGrow factorization on prebuilt, row-sharded
  matrices using documented PyTorch/NCCL configurations.

The following are outside the current support scope:

- Financial strategies, factor definitions, portfolio construction, or advice.
- Production data ingestion and proprietary datasets.
- Distributed rank selection, streaming matrix construction, and distributed
  spherical K-means.
- Unlisted dependency, hardware, or container combinations.

## Requesting Help

Before opening an issue:

1. Review the README and relevant documents under `docs/`.
2. Reproduce the issue using synthetic data when possible.
3. Record the code revision, Python/PyTorch/CUDA/NCCL versions, GPU model,
   container image, precision mode, and complete command.

Use NVIDIA Product Security channels described in `SECURITY.md` for
vulnerabilities rather than opening a public issue.
