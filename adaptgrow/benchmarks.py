# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate benchmark matrices at multiple sizes and seeds.

Writes to data/benchmarks/<type>_p<size>_s<seed>.dat with companion
files (.meta.json, .perm.npy, .labels.npy).

Use ``scripts/generate_benchmarks.py`` from a repository checkout.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from .corr_construction import corr_construct
from .tpdm_construction import tpdm_construct

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "benchmarks"

DEFAULT_SIZES = [100, 1_000, 10_000]
DEFAULT_SEEDS = [42, 99, 7]
MATRIX_BUILDERS = {
    "tpdm": (tpdm_construct, {}),
    "corr": (corr_construct, {}),
}


def generate_all(sizes, seeds, output_dir, matrix_types):
    """Generate requested synthetic matrices and their companion metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in sizes:
        for seed in seeds:
            for mat_type in matrix_types:
                construct_fn, kwargs = MATRIX_BUILDERS[mat_type]
                name = f"{mat_type}_p{p}_s{seed}"
                path = output_dir / f"{name}.dat"

                if path.exists():
                    print(f"  skip {name} (exists)")
                    continue

                print(f"  generating {name} ...", end="", flush=True)
                t0 = time.time()
                Sigma, perm, labels = construct_fn(
                    output_path=str(path), p=p, random_state=seed, **kwargs
                )
                elapsed = time.time() - t0

                size_mb = path.stat().st_size / 1e6
                print(f"  {elapsed:.1f}s  {size_mb:.1f} MB  "
                      f"diag_ok={np.allclose(np.diag(np.asarray(Sigma[:10, :10])), 1.0)}  "
                      f"k={int(np.floor(np.sqrt(p)))}  "
                      f"labels={sorted(set(labels))[:5]}...")

    # Summary
    total_bytes = 0
    n_files = 0
    for path in output_dir.iterdir():
        if path.is_file():
            total_bytes += path.stat().st_size
            n_files += 1
    print(f"\nDone. {n_files} files, {total_bytes / 1e9:.2f} GB total in {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic correlation and TPDM benchmark matrices."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--matrix-types",
        nargs="+",
        choices=sorted(MATRIX_BUILDERS),
        default=sorted(MATRIX_BUILDERS),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    generate_all(args.sizes, args.seeds, args.output_dir, args.matrix_types)


if __name__ == "__main__":
    main()
