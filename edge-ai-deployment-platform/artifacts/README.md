# Release and qualification evidence

- `coverage.json` — machine-readable coverage for the current working tree. The configured gate uses two-decimal precision and requires >=90.00% combined statement+branch coverage.
- `qualification.json` — current local qualification snapshot, including test count, coverage, native checks, package hash, reference evaluation, and the public-data showcase boundary.
- `sbom.spdx.json` — SPDX 2.3 runtime/service dependency graph generated from project metadata and the observed qualification environment.
- `qualification-constraints.txt` — exact qualification-environment package versions. This is an environment record, not a cross-platform hash-locked resolver file.

The package version remains `1.0.0` until a maintainer deliberately cuts a release. Qualification artifacts therefore use `release: unreleased` when they describe the evolving working tree.

The UCI HAR raw dataset is intentionally not stored here. A network-enabled showcase run writes acquisition provenance beside the generated experiment evidence; see `DATASETS.md` and `docs/showcase/EXPERIMENTS.md`.
