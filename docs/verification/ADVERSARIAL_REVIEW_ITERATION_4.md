# Adversarial Review — Iteration 4

## Executive assessment

Iteration 4 moves the project from bounded *admission* toward bounded *execution*. The strongest new control is an optional spawned-process inference supervisor that can terminate and restart a backend which hangs or crashes. The production engine bootstraps that child from bounded bytes taken from the exact authenticated active-slot inode plus the signed model kind and ABI; it does not transfer a live runtime/session object across the process boundary.

This materially improves availability containment, but it does not turn the repository into a fully qualified device runtime. The highest remaining risks have shifted toward deployment-time backend isolation, real device identity and hardware trust, target-specific resource control, real ONNX/accelerator parsers, and server/kernel ingress.

## Steelman attacks and resulting changes

### 1. A bounded queue does not bound a backend after it starts

**Attack:** submit an input that makes a native backend block indefinitely. Queue-start deadlines have already been satisfied, and Python cannot safely cancel the worker thread.

**Change:** add `ExecutionMode.PROCESS`. One spawned child belongs to each executor worker. Actual inference has its own hard timeout; timeout terminates the child and the next request starts a clean child. Deliberate child crashes are contained and recovered.

### 2. Pickling a live runtime is not a production isolation contract

**Attack:** future ONNX Runtime/TensorRT sessions may not be pickleable, may carry native pointers, or may deserialize in ways that expand the trust boundary.

**Change:** the parent re-authenticates the active slot and sends bounded immutable model bytes plus signed `model_kind`, `runtime_abi` and generation. The child constructs its own runtime. The runtime-object path remains compatibility-only for tests/custom engines.

### 3. A malicious backend can amplify a small request into a huge response

**Attack:** return a gigantic class dimension, wrong batch, strange dtype or non-finite tensor and force parent allocation/JSON serialization or trigger internal errors.

**Change:** the engine now enforces maximum output classes and total output elements, exact rank/batch semantics, numeric representability and finite values. The child performs an independent bound check before IPC serialization, and the parent validates again before prediction/HTTP serialization.

### 4. Killing a hung child does not stop a fast memory runaway

**Attack:** allocate aggressively enough to pressure the host or trigger the OOM killer before the execution timeout expires.

**Change:** process mode has an opt-in `process_memory_limit_mb`. On the Linux/POSIX qualified path the child applies `RLIMIT_AS` before runtime load. The setting is intentionally not given a production default because virtual-address-space behavior differs substantially across BLAS, ONNX, CUDA/TensorRT and vendor runtimes.

**Residual:** cgroups/systemd/container limits and target-specific worker-count/model-memory budgets remain necessary. `RLIMIT_AS` is only one containment layer.

### 5. Readiness can lie after an idle worker dies

**Attack:** kill a child while no inference is active. A readiness endpoint that checks only deployment state can remain green until the next request fails.

**Change:** `/readyz` verifies executor health and repairs an idle dead child. Executor metrics expose starts, restarts, crashes, execution timeouts and live process-worker count.

### 6. Queue timeout and execution timeout are operationally different failures

**Attack:** treat both as generic 504s and operators cannot distinguish overload from a wedged backend.

**Change:** HTTP responses distinguish queue-start timeout from execution timeout with `X-EdgeAI-Timeout`, and metrics separately count execution timeouts.

### 7. HTTP body bounds can still accept ambiguous framing at the app boundary

**Attack:** duplicate `Content-Length`, simultaneous `Content-Length` and `Transfer-Encoding`, malformed lengths, or a body whose observed size disagrees with its declared length.

**Change:** ASGI middleware rejects ambiguous framing, uses strict decimal `Content-Length` grammar, and checks the final observed byte count where a content length is present.

**Residual:** the web server, reverse proxy and kernel parse the network request before this middleware. Production must still qualify those layers against request smuggling, slow-client and connection-exhaustion threats.

### 8. SBOM/reproducibility evidence can silently drift

**Attack:** keep a committed SBOM with a stale hard-coded timestamp/dependency list while `pyproject.toml` changes, creating attractive but false supply-chain evidence.

**Change:** SBOM generation derives the project/version/runtime dependency roots from project metadata, supports deterministic `SOURCE_DATE_EPOCH`, and CI regenerates SBOM/qualification constraints and fails on drift. CI also builds the wheel twice with the same epoch and byte-compares the outputs.

## What is strongest now

- canonical signed artifact identity and same-handle verification/load;
- crash-recoverable A/B lifecycle with anti-rollback and optional external anchor protocol;
- strict signed/config parsing;
- bounded request admission *and* optional killable backend execution;
- independent input and output tensor resource contracts;
- retry-stable offline telemetry;
- adversarial fault-injection coverage for process startup, IPC, cancellation, timeout and crash paths;
- reproducibility checks tied to source metadata rather than static documentation.

## Highest-value remaining work

### P1 — isolate deployment-time model loading and health benchmarking

Candidate load/benchmark still runs in the lifecycle process. A future complex parser/backend can hang, crash, reserve excessive memory, or make `ru_maxrss` reflect unrelated historical process peaks. Move candidate qualification into a dedicated killable process with explicit validation-dataset bounds, wall-clock deadline and per-candidate memory accounting.

### P1 — real external trust and device identity

Implement and physically qualify a TPM/secure-element monotonic anchor, then add device/service identity, mTLS, protected private-key storage and a signed trust-store rotation protocol. The software anchor transaction is not proof of hardware durability.

### P1 — real production backend

Integrate ONNX Runtime CPU first. Define accepted operator/model/schema policy, external-data rules, memory/thread/session options, parser limits and deterministic health tests before TensorRT/CUDA. The process bootstrap was designed so native sessions can be constructed in their owning child.

### P1 — target-specific resource containment

Qualify worker count, `RLIMIT_AS`/cgroups, CPU affinity, native thread pools, filesystem quotas and restart policy on the real device. Multiple process workers intentionally multiply runtime memory.

### P1 — network/server boundary

Pin and test Uvicorn/proxy/server connection limits, keep-alive/read timeouts, request-buffering behavior, TLS handshake limits and multi-process aggregate capacity. ASGI checks are not a substitute for network admission control.

### P2 — scalable process bootstrap

For large models, avoid copying full model bytes through a multiprocessing pipe. Prefer a parent-authenticated file descriptor, sealed shared memory/memfd, or backend-specific immutable model bundle with explicit lifetime and integrity semantics.

### P2 — release provenance and dependencies

Pin container bases by digest, decide whether GitHub Actions should be commit-SHA pinned, generate signed provenance/attestations and formalize SBOM license/vulnerability handling. Reproducible local wheel bytes are useful evidence but not full supply-chain provenance.

### P2 — telemetry integrity

Retry IDs solve idempotency, not tampering. If telemetry becomes audit/security evidence, add sequence/integrity semantics and a protected signing/MAC key strategy.

## Language decision

Retain Python + C++20. Process isolation closes the hard-deadline problem without adding a third production language. Python remains appropriate for lifecycle orchestration, API, cryptographic-policy integration, tests and qualification. C++ remains the natural boundary for high-performance kernels and future ONNX/TensorRT/vendor integration. Rust or Go should be introduced only if a concrete privileged-agent or memory-safety requirement justifies the additional build, supply-chain and qualification surface.

## Qualification boundary

Iteration 4 qualifies the reference process supervisor, authenticated artifact bootstrap, output bounds, ASGI framing checks, optional POSIX address-space limit plumbing, and reproducible-evidence workflow in the local Linux environment. It does **not** qualify ONNX Runtime, TensorRT/CUDA, a real TPM, mTLS, target hardware, production cgroups/server configuration, or physical power-loss/thermal behavior.
