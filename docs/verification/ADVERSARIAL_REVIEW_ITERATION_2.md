# Adversarial Steelman Review — Iteration 2

Date: 2026-08-20

## Executive conclusion

The second hardening iteration moves the repository from “qualified security primitives” toward a coherent single-device reference agent. The three previous P1 software gaps are now implemented: bounded HTTP inference admission, cooperative inter-process filesystem locking, and pre-allocation NPZ model inspection. The deployment state machine is also materially stronger against crash residue and time-of-check/time-of-use races.

The strongest remaining objections are now mostly **platform and operational trust boundaries**, not missing Python plumbing: rollback of the whole filesystem, device identity/TLS, hostile-root behavior, target-hardware timing/power/thermal qualification, production backend integration, telemetry audit semantics, and release provenance.

## New invariants established

### 1. Integrity verification and model loading share the same inode

A verified registry/slot model is opened with no-follow semantics, hashed through a stable file descriptor, preflighted, and loaded from that same handle. Replacing the pathname after verification no longer changes the bytes consumed by NumPy.

**Why this matters:** path-level “verify, close, reopen, load” is a classic TOCTOU shape. A signature is only useful if it authenticates the bytes actually parsed and executed.

### 2. Signed semantic version is not treated as artifact identity

A slot manifest must equal the canonical signed manifest stored for that version in the immutable registry. A different artifact that is also validly signed but reuses the same semantic version is rejected.

**Residual assumption:** a privileged actor who can replace the canonical registry itself with another validly signed same-version release is outside the application-only trust model. Release governance should enforce version uniqueness and publish provenance/transparency evidence.

### 3. Compressed NPZ structure is inspected before NumPy allocation

The loader checks exact member names, duplicate/nested/encrypted/unsupported members, declared expanded bytes, NPY version/header size, dtype, rank/shape, and model-specific dtype/shape contracts before calling `numpy.load`. Object arrays are forbidden. A configurable expanded-size ceiling defaults to 256 MiB.

**Why this matters:** signed-but-malformed or compromised-signer artifacts should not get an unlimited allocation opportunity merely because their compressed file is small.

### 4. Registry/deployment loading uses no-follow regular-file opens

Registry metadata and model bytes are opened relative to a stable directory descriptor. Model handles are not reopened between hashing and model parsing. Deployment state, model artifacts, lock files, and telemetry spool paths reject final-component symlinks where they are security-sensitive.

### 5. Deployment recovery distinguishes commit boundaries

Recovery now authenticates `.backup-*` directories against durable state, restores a pre-commit backup when appropriate, discards a stale post-commit backup, removes an uncommitted first-write slot, and rejects slot release sequences above the persisted anti-rollback floor.

**Why this matters:** “A/B + atomic state file” is insufficient if crash residue is ambiguous after restart.

### 6. Runtime cache identity is deployment generation plus version

The inference engine no longer assumes semantic version uniquely identifies an in-memory runtime. Cache refresh obtains `(runtime, version, generation)` as one deployment snapshot under a shared process lock.

### 7. Public inference admission is actually bounded

`/v1/predict` is async and submits to a fixed-worker executor. An outer in-flight middleware rejects excess requests before body consumption; the internal queue bounds waiting work. Queue saturation maps to HTTP 429, and queue-start deadline expiry maps to 504. Drop policies deterministically complete affected callers.

**Residual boundary:** Uvicorn/OS socket acceptance and backlog occur before ASGI middleware. Production launch configuration must still bound server-level concurrency/connections, and aggregate capacity must be considered if multiple web processes are used.

### 8. Authentication precedes prediction body allocation

Raw-header API-key middleware is outermost for protected endpoints. Unauthorized prediction requests are rejected before request-body parsing. The application body limiter then bounds both declared and chunked bodies.

**Residual boundary:** API keys are a reference control, not device identity. Real deployments should terminate authenticated TLS/mTLS and rotate secrets using a platform trust store.

### 9. Cooperative multi-process mutation is serialized

Registry add, deployment transitions, and telemetry spool rewrites use POSIX advisory locks. Regression tests exercise lock blocking and concurrent telemetry appends across spawned processes.

**Residual boundary:** advisory locks only protect cooperating processes. A hostile privileged actor or code that ignores the protocol can still mutate files.

### 10. Telemetry no longer follows a spool symlink

Spool and lock file opens use no-follow/regular-file checks. This closes an avoidable local path-redirection weakness. Telemetry event integrity itself remains outside the current qualified boundary.

## Strongest remaining adversarial arguments

### A. Full-filesystem rollback still defeats application anti-rollback

The durable floor lives on the same rollbackable filesystem as deployment state. A hypervisor snapshot, disk image restore, or privileged physical adversary can rewind both state and slots together.

**Next control:** TPM/secure element/bootloader monotonic counter or another external monotonic trust anchor.

### B. A trusted signer can still authorize a resource-hostile model within configured bounds

NPZ preflight caps expanded bytes and validates the reference linear schema, but it does not establish application safety, latency, or device-specific memory behavior for arbitrary future backends.

**Next control:** give every new backend its own format preflight contract, allocator/resource limits, and target-device qualification. Do not route ONNX/TensorRT through a generic “trusted file” exception.

### C. HTTP application limits are not network admission control

ASGI middleware executes after the server and kernel have accepted work. Slowloris behavior, connection floods, TLS handshakes, socket backlog, and proxy buffering are not solved here.

**Next control:** qualify explicit server/proxy settings and, for hostile networks, put the service behind a device gateway or process-isolated ingress with connection limits.

### D. Inference has queue-start deadlines, not hard execution deadlines

Once NumPy/native inference begins, Python does not cancel the worker thread. A hung backend can occupy a worker indefinitely and can also delay graceful shutdown.

**Next control:** production backends should provide bounded execution, watchdog/reset support, or inference process isolation where hard deadlines are safety/availability requirements.

### E. Telemetry is durable and bounded but not an audit log

Counters are process-local; a crash after an external sink side effect but before spool rewrite can duplicate an event; local edits are not cryptographically detectable.

**Next control:** persistent event IDs/idempotency keys, device identity, and optional hash chaining/signatures if telemetry is used for audit/security decisions.

### F. Metrics do not prove target-device real-time behavior

`ru_maxrss` is a process high-water mark, and the reference latency harness is not WCET, thermal, power, throttling, or concurrent-load evidence.

**Next control:** cold/warm distributions, concurrency profiles, backend-internal thread controls, thermal soak, power/current capture, restart and power-cut campaigns on the intended hardware.

### G. Release provenance remains weaker than runtime artifact verification

The runtime authenticates model releases well, but container base tags can still be mutable and SBOM inventory is not a provenance attestation.

**Next control:** pin OCI base images by digest, publish signed build provenance/attestations, define trust-store update signing, and retain release approval records.

## Language decision after iteration 2

No new production language is justified. Keep:

- **Python 3.11+** for lifecycle orchestration, FastAPI, qualification, reference inference, and research.
- **C++20** for native kernels and future ONNX Runtime/TensorRT/vendor SDK adapters.

Adding Rust/Go/TypeScript now would create more supply-chain and qualification surface without addressing the remaining platform controls. Revisit a memory-safe dedicated device daemon only if process isolation, long-lived fleet connectivity, or hard operational requirements make Python unsuitable.

## Priority order for iteration 3

1. External monotonic anti-rollback anchor abstraction with a software test double and TPM/secure-element adapter contract.
2. Device identity/mTLS and signed trust-store/key rotation format.
3. ONNX Runtime CPU backend with explicit tensor/shape/allocation preflight and the same stable-handle trust contract where supported.
4. Production server launch profile: bounded connections/concurrency, timeouts, native thread-pool controls, graceful-shutdown policy.
5. Persistent telemetry event IDs/idempotency and optional tamper evidence.
6. Target-device fault campaign: process kill at transaction boundaries, filesystem full, permissions, restart, power-loss, long soak.
7. Reproducible release provenance and OCI digest pinning.

The project should continue to resist “feature theater”: fleet UI, multi-model routing, and accelerator marketing are lower value than proving these controls.
