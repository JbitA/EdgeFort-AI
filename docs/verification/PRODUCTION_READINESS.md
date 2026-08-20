# Production readiness boundary

## Qualified in this repository

The current tree locally qualifies the following software properties:

- strict Ed25519 signed-release identity, key revocation and immutable registry semantics;
- stable-handle artifact verification/load and bounded NPZ preflight;
- signed model-kind/runtime-ABI binding;
- crash-consistent, health-gated A/B deployment and authenticated rollback recovery;
- application anti-rollback plus an optional external-anchor crash-transaction protocol using software test doubles;
- isolated deployment-time candidate parsing/benchmarking with wall-clock and validation-data limits;
- bounded authenticated FastAPI prediction admission;
- bounded queues, queue-start deadlines, tensor element limits and backend output validation;
- optional killable process-isolated inference with safe non-pickle IPC;
- backend child environment stripping, core-dump suppression, `no_new_privs`, file-descriptor cap and executor-wide restart circuit breaker;
- strict runtime configuration parsing;
- durable offline telemetry with retry-stable event IDs;
- PSI drift monitoring;
- narrow ONNX Runtime CPU adapter contract tests;
- safe/provenance-aware public UCI HAR data acquisition logic and reproducible showcase generation;
- >=90.00% combined statement+branch coverage gate at two-decimal precision;
- C++20 Release and ASan/UBSan qualification;
- offline wheel build and installed CLI smoke path;
- CodeQL, Dependabot, multi-Python CI, reproducible wheel and SBOM/lock workflow definitions.

## Current qualification snapshot

See `PROJECT_STATUS.json`, `artifacts/qualification.json`, and `artifacts/coverage.json` for machine-readable evidence. The current local snapshot records 301 passing Python tests and 90.50% combined coverage.

## Important distinctions

### Public-data showcase

The UCI HAR acquisition and experiment path is implemented and tested against a schema-faithful local ZIP fixture. This execution environment cannot fetch the upstream 58 MB archive, so **real UCI HAR model metrics are not claimed in the local qualification snapshot**. The manual GitHub workflow is designed to produce that evidence in a network-enabled runner.

### External rollback anchor

`MemoryRollbackAnchor` and `FileRollbackAnchor` prove the software protocol only. Whole-filesystem rollback resistance requires a protected monotonic implementation such as TPM NV state, secure-element state, or a verified boot-chain service plus target power-loss qualification.

### Process isolation

Process workers improve fault containment and provide hard execution deadlines. They are not a complete OS sandbox while parent and child share the same service principal. Production deployment still needs target-specific cgroup/service-manager/namespace/seccomp decisions.

### ONNX Runtime

The `onnxruntime-cpu` adapter intentionally accepts a narrow single-input/single-output fixed-shape float32 contract. Until actual ORT is installed and subjected to parser, memory, timeout, crash, health-gate, and inference qualification, the repository does not claim native ONNX Runtime readiness.

### Network bounds

ASGI/request limits do not cap TLS/socket work already accepted by the server or kernel. Production ingress settings require separate qualification.

### Target hardware

All committed latency/RSS measurements are host-reference evidence. They do not establish Jetson/mobile/industrial power, thermal, accelerator, latency, or WCET behavior.

## Required before claiming a real device release

1. Execute and retain the public UCI HAR showcase in a clean network-enabled release environment.
2. Qualify real ONNX Runtime CPU under the isolated deployment/inference boundaries.
3. Run the backend under a distinct security principal with explicit filesystem/network/cgroup/seccomp or equivalent restrictions.
4. Implement and power-fault-test a real protected monotonic rollback anchor.
5. Add device identity, mTLS, protected credentials, and signed trust-store rotation.
6. Define and test web-server/kernel/proxy ingress limits.
7. Qualify one named edge target for latency, memory, power, thermal behavior, restart and soak.
8. Pin release container/base images by digest and publish signed provenance/attestations.
9. Define production signing-key custody/rotation and recovery procedures.
10. Add application-specific safety/fallback analysis where the deployment domain requires it.

The project remains a production-oriented reference implementation and qualification harness, not a regulatory certification or complete fleet-management product.
