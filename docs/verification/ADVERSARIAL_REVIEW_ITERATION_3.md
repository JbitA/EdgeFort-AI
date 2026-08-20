# Adversarial Steelman Review — Iteration 3

Date: 2026-08-20

## Executive conclusion

Iteration 3 moves the project from application-only rollback policy toward a design that can bind release monotonicity to a device trust primitive without sacrificing crash recovery. It also removes parser ambiguity from signed manifests/configuration, gives telemetry retry-stable idempotency identifiers, and prevents graceful service shutdown from waiting forever on a wedged worker.

The largest remaining objections are now **target integration and isolation**: no real TPM/secure-element adapter is qualified, device identity/mTLS is absent, running native inference still cannot be forcibly stopped inside a Python thread, network admission remains outside ASGI, and no ONNX Runtime/accelerator or target-hardware campaign has been qualified.

## New invariants established

### 1. External monotonic rollback anchor is a first-class optional boundary

`RollbackAnchor` defines a minimal adapter contract. The deployer can compare local release state to an external monotonic floor and refuses local state that is behind/ahead in an unexplained way.

The included memory/file implementations are test doubles; the file implementation is deliberately documented as **not** protection from whole-filesystem rollback.

### 2. Anchor-first deployment is crash recoverable

The security-preserving order is journal -> external anchor -> local state. A pending journal plus canonical signed slot identity lets startup complete a crash that occurred after the irreversible anchor update but before `state.json` commit. Missing/conflicting recovery evidence fails closed.

### 3. Release sequence is unique release identity

The immutable registry rejects reuse of one `release_sequence` by two semantic versions and fails closed if existing published entries are corrupt while it is trying to establish uniqueness. This prevents external-anchor recovery from having multiple valid signed interpretations for one monotonic value.

### 4. Signed manifest parsing is non-coercive and unambiguous

The manifest parser now rejects unknown/missing fields, duplicate JSON object keys, booleans in integer/numeric fields, string-to-number coercion, non-finite constants, invalid metadata shape, and release counters outside the supported range.

**Why this matters:** a signature should authenticate exactly one typed schema. Permissive coercion can let a syntactically altered manifest verify as the same reconstructed object and can hide unsigned/confusing fields from operators or downstream tooling.

### 5. Configuration parsing follows the same fail-closed principle

Runtime YAML rejects duplicate keys, unknown keys, boolean-as-integer values, non-finite numeric limits, invalid root/API-key variable strings, and missing required fields. Configuration can no longer silently change meaning through YAML type quirks such as `true == 1` in Python.

### 6. Telemetry retries carry stable idempotency identifiers

New spool records contain a 128-bit persistent event ID. `flush()` remains backward-compatible and emits only the original event. `flush_records()` emits the stable ID with the event so a remote sink can deduplicate a retry when its side effect succeeded but the acknowledgement/local spool rewrite did not.

Caller-supplied event IDs also make local enqueue idempotent and reject reuse of one ID for different content. Legacy raw JSONL records remain readable but are explicitly marked as lacking persistent IDs.

### 7. Graceful shutdown is bounded

`BoundedInferenceExecutor.close()` accepts a timeout and reports whether all daemon workers stopped. FastAPI lifespan uses a bounded shutdown wait. This prevents a hung backend from indefinitely blocking graceful process shutdown.

**Important:** this is not hard execution cancellation. The running worker is not asynchronously killed.

## Strongest remaining adversarial arguments

### A. No production rollback anchor adapter is qualified yet

The software protocol, crash transaction, ambiguous-ack poison state, and cooperative peer refresh are implemented, but `MemoryRollbackAnchor` and `FileRollbackAnchor` do not prove resistance to disk-image rollback. A TPM/secure-element/boot-chain adapter must be implemented and tested on the intended device, including power-cut behavior and counter exhaustion/provisioning.

### B. Long-running process observation of out-of-band anchor changes needs a defined operational model

The deployer caches the configured anchor floor to avoid a TPM/secure-element read on every inference. It now refreshes the anchor when durable local state changes under a cooperating peer or while a pending anchor journal exists, and a process whose own anchor advance returns ambiguously is poisoned until fresh recovery. This covers normal cooperative lifecycle transitions without putting hardware latency on the hot path. It is still not a continuous hardware-anchor watcher: an anchor changed outside the deployment protocol, followed by rollback of all local state/journal evidence before observation, requires lifecycle-agent ownership or a qualified periodic/watch policy.

### C. Native inference still lacks a hard watchdog boundary

Bounded queue-start deadlines and bounded shutdown waiting do not interrupt a C/C++/vendor call after it begins. A backend deadlock can consume a worker indefinitely.

**Next control:** process-isolated inference workers or a backend/device watchdog with reset semantics; qualify failure/restart behavior on target hardware.

### D. API keys are not device identity

Raw-header authentication protects the reference HTTP boundary from unauthenticated body allocation, but API keys do not provide certificate-bound device identity, attestation, secure key storage, or fleet rotation.

**Next control:** mTLS/device certificates backed by platform key storage and a signed trust-store rotation procedure.

### E. Network admission precedes ASGI

Socket acceptance, TLS handshakes, proxy buffers, slow clients and kernel backlog are outside the bounded application queue.

**Next control:** publish and qualify an explicit Uvicorn/reverse-proxy/systemd launch profile with connection/concurrency/timeouts and hostile-network tests.

### F. Telemetry is idempotency-capable but not tamper evident

Stable IDs solve duplicate side effects when the sink honors them, but local telemetry records can still be edited/deleted by a privileged actor and there is no cryptographic sequence/hash chain.

**Next control:** device-bound signing or hash chaining only if telemetry will be used as security/audit evidence; otherwise keep it operational telemetry and avoid overstating integrity.

### G. Production backend/parser risk remains unqualified

The NPZ reference parser has strong pre-allocation checks. ONNX, TensorRT engines and vendor formats have different parsers, external data files, allocator behavior and native thread pools.

**Next control:** ONNX Runtime CPU integration first, with explicit model format/shape/resource contract and isolated benchmarks, before accelerator work.

### H. Release provenance remains incomplete

Runtime artifact authentication does not prove how the Python wheel/container/model was built. The Docker base image is not yet pinned by digest and build provenance is not signed.

**Next control:** digest pinning, deterministic lock/constraint policy, SBOM refresh in CI, and signed build attestations/provenance.

## Recommended Iteration 4 order

1. Implement a **real adapter boundary** for TPM/secure-element monotonic state behind the existing protocol, with a fake device emulator for fault injection and a target-specific integration document.
2. Add process-isolated inference execution with kill/restart deadlines for backends that cannot guarantee bounded calls.
3. Add device identity/mTLS and a signed trust-store/key-rotation format.
4. Integrate ONNX Runtime CPU with backend-specific preflight and resource/thread limits.
5. Publish/qualify production server + systemd/reverse-proxy settings for network admission and graceful restart.
6. Add filesystem-full, permission-loss, SIGKILL-at-boundaries and target power-cut campaigns.
7. Pin OCI bases by digest and publish signed release provenance.
8. Add telemetry hash chaining/signatures only if product requirements promote telemetry to audit evidence.

Fleet dashboards, multi-model routing, GPU marketing and extra languages remain lower-value than these controls.
