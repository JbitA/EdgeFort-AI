# ADR 0003: Python control/reference plane and C++20 native runtime boundary

## Status

Accepted.

## Context

The project needs fast iteration and strong testability for artifact trust, deployment state, qualification, research, and service behavior, while also needing a path to vendor inference SDKs and performance-critical edge kernels.

Adding languages increases the trusted code base, build matrix, FFI surface, dependency graph, and qualification burden.

## Decision

Use **Python 3.11+** for the lifecycle/control plane, service boundary, reference runtimes, evaluation, and tests. Use **C++20** for native math kernels and future device/backend integrations where the vendor ecosystem or performance requirement justifies native code.

Do not add a third production language without a new ADR that demonstrates a concrete requirement not adequately served by these boundaries.

## Rationale

- Python provides the strongest productivity-to-verifiability ratio for the current lifecycle problem and ML/reference ecosystem.
- C++ is the native integration language for common edge inference stacks including ONNX Runtime C/C++ APIs, TensorRT, CUDA, and many vendor SDKs.
- A Rust device agent may become attractive for memory-safe systems programming, but adding Rust before such a daemon exists would duplicate architecture rather than reduce risk.
- Go/TypeScript do not solve a current core requirement.

## Consequences

- Native functionality should have a narrow interface and independent tests/sanitizer coverage.
- Control-plane correctness must not depend on a native extension when a Python reference implementation is sufficient for qualification.
- Any Python/native behavior that should be equivalent (for example integer overflow semantics) requires cross-backend conformance tests.
