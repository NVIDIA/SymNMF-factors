# Contributing

Thank you for your interest in SymNMF-factors. All contributors must follow
`CODE_OF_CONDUCT.md`.

## Before You Start

- Open an issue before implementing a large solver, interface, dependency, or
  benchmark-format change.
- Keep the project independent of private infrastructure and proprietary data.
- Use synthetic or otherwise redistributable data in tests and examples.
- Do not add financial strategies, factor definitions, portfolio rules, or
  trading logic.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q -m "not gpu and not distributed"
```

Use the public NGC PyTorch environment described in `docs/runtime.md` for CUDA
and distributed work.

## Testing

Add or update tests for behavioral changes:

```bash
# Required CPU suite
pytest -q -m "not gpu and not distributed"

# Optional hardware suites
pytest -q -m "gpu and not distributed"
pytest -q -m distributed
```

Changes to solvers, generators, or benchmarks must state their seed, matrix
parameters, precision/TF32 behavior, device assumptions, and relevant runtime
versions.

## Pull Requests

A pull request should:

- Explain the problem and why the change is needed.
- Keep changes focused and reviewable.
- Include tests and documentation where applicable.
- Report the validation commands and results.
- Avoid generated data, notebook execution artifacts, credentials, and
  machine-specific paths.
- Update `CHANGELOG.md` for user-visible changes.

Maintainers review contributions for correctness, scope, maintainability,
security, reproducibility, and compatibility with supported configurations.

## Signing Your Work

This project uses the Developer Certificate of Origin (DCO). Sign off every
commit with the `-s` flag:

```bash
git commit -s -m "Describe your change"
```

This adds a `Signed-off-by` line certifying that you agree to the DCO below.
Use your real name and an email address associated with your contribution.

### Developer Certificate of Origin

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
