#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Thin multi-node launcher for run_distributed.py.
#
# One task per GPU. We export RANK / WORLD_SIZE / LOCAL_RANK from SLURM and use
# torch's env:// rendezvous, so no torchrun is needed and the exact same script
# runs on one node or many (only --nodes changes). Submit with:
#
#     MATRIX_DIR=/path/to/S_windows sbatch scripts/launch.slurm
#
#SBATCH --job-name=symnmf_dist
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4          # = GPUs per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=symnmf_dist_%j.out

set -euo pipefail

# Rendezvous: first node in the allocation is the master.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export MATRIX_DIR="${MATRIX_DIR:?set MATRIX_DIR to the directory of pre-built S_t matrices}"

srun --kill-on-bad-exit=1 bash -c '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  python scripts/run_distributed.py \
    --matrix-dir "$MATRIX_DIR" \
    --kind corr \
    --k 8 \
    --io-method auto
'
