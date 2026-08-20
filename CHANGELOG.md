# Changelog

## Unreleased

- add optional external monotonic rollback-anchor protocol with a crash-recoverable journaled commit;
- reject release-sequence reuse across registry versions so anchor values have unique canonical release identity;
- make signed-manifest JSON parsing strict and non-coercive, including duplicate-key rejection;
- make runtime YAML configuration reject duplicate keys and boolean/numeric ambiguity;
- add retry-stable telemetry event IDs, idempotent enqueue, record-aware flush, and legacy spool compatibility;
- bound graceful inference-worker shutdown without unsafe asynchronous thread cancellation;
- expand adversarial regression coverage to 194 tests while retaining the >=90% branch-aware coverage gate;
- add Iteration 3 ADR, threat-model, readiness, runbook and traceability updates.
- re-verify staged and active slot artifacts at the execution boundary;
- enforce the signed runtime ABI against the selected model runtime;
- strictly validate deployment state and fail closed on missing state with populated slots;
- reject registry symlink/path-manifest mismatches and re-verify staged registry copies;
- harden manifest timestamp, digest, metadata, and trusted-key validation;
- add API request-body limits, production-docs suppression, and ragged-input normalization;
- add dynamic-int8 accumulator bounds and native overflow regression tests;
- preserve telemetry ordering after delivery failures and make spool trimming linear;
- add adversarial regression tests, CodeQL, expanded CI/package/container checks, and a security-aware PR template;
- document the Python/C++ language boundary and adversarial production-readiness backlog.

## 1.0.0

- asymmetric Ed25519 artifact signing and key revocation;
- immutable registry and runtime ABI metadata;
- A/B health-gated deployment with anti-rollback release floor;
- state-write fault recovery and deterministic rollback;
- float, weight-int8 and dynamic-int8 reference runtimes;
- bounded inference queue and durable offline telemetry;
- authenticated inference API and runtime metrics;
- PSI drift monitoring;
- C++20 native int8 primitive with sanitizer qualification;
- branch-aware coverage and multi-seed robustness evaluation;
- reproducible packaging, SBOM and release evidence.

## Unreleased — GitHub showcase evolution

- Reframed the repository front page around the operational value of secure edge-model lifecycle controls rather than the reference classifier.
- Added a GitHub-renderable architecture diagram and concise proof/failure matrix.
- Added explicit, size-bounded and ZIP-hardened acquisition of the CC BY 4.0 UCI Human Activity Recognition Using Smartphones dataset with DOI/license/source provenance and observed upstream SHA-256 recording.
- Added a reproducible `edgeai showcase` experiment that measures float/int8 trade-offs, PSI response under a bounded feature-space shift proxy, signed-but-bad release rejection, A/B rollback and anti-rollback replay rejection.
- Added generated JSON, SVG and Markdown showcase evidence plus a fully offline digits smoke track.
- Added `make data-uci-har`, `make showcase`, and `make showcase-real` targets.
- Added a manually triggered GitHub `showcase-real-data` workflow so public-data evidence can be regenerated without committing the raw dataset.
- Added 21 showcase/data/CLI tests; current full suite is 301 passing tests.
- Regenerated strict coverage evidence at 90.50% combined statement+branch coverage (91.77% statement, 86.33% branch-hit) with two-decimal gating.
- Refreshed project status, production-readiness, maturity, architecture, and current adversarial-review documentation to remove stale iteration claims.
