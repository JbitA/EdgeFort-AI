# Current adversarial steelman review

Date: 2026-08-20

## Executive assessment

This repository is now a strong **edge-AI deployment systems showcase and qualification harness**. Its value is not the linear reference classifier; it is the explicit treatment of model authenticity, semantic health, crash-consistent A/B activation, anti-rollback, bounded admission, isolated execution, offline telemetry, drift, and reproducible evidence.

The strongest case for the project is that it demonstrates controls that ordinary `model.predict()` demos omit, and it preserves negative results and qualification limits rather than marketing around them.

The strongest skeptical case is that several remaining guarantees still stop at the application/process boundary. Real TPM/secure-element behavior, device identity, OS-level sandboxing, network-ingress hardening, real native ONNX qualification, and target hardware behavior are not yet proven.

## Current strengths

### Release trust and identity

- Ed25519 detached signatures and explicit trusted/revoked public keys.
- Strict canonical manifest schema with duplicate/unknown/coercion rejection.
- Exact model SHA-256 and byte size bound to the signed manifest.
- Unique monotonic release sequences and canonical release identity in the immutable registry.
- Runtime ABI/model-kind compatibility is signed and enforced.

### Verification where bytes are consumed

- Registry and deployment paths reject symlink/path confusion.
- Model verification and loading share stable no-follow handles for the qualified NPZ path.
- Staged, renamed, rollback-target, and active-slot artifacts are re-authenticated.
- NPZ structure/header/expanded-size constraints are enforced before NumPy allocation.

### Deployment and rollback

- Candidate releases are qualified before promotion into the inactive A/B slot.
- State changes are atomic and crash residue is authenticated during recovery.
- Operator rollback does not lower the anti-rollback release floor.
- An optional external rollback-anchor transaction uses a durable journal to recover the anchor-first commit window.
- Deployment-time model parsing/benchmarking can execute in a fresh killable process with a wall-clock bound and cleaner per-candidate RSS accounting.

### Runtime containment

- HTTP admission is bounded before expensive model work.
- Inference queues, request bytes, tensor dimensions/total elements, and backend outputs are bounded.
- Optional process workers use a versioned raw-byte protocol rather than Python pickle.
- Backend processes are stripped of inherited environment variables before backend loading, core dumps are disabled, Linux `no_new_privs` is applied, and file-descriptor growth can be capped.
- Hung/crashed workers are killable/restartable and an executor-wide rolling restart budget prevents concurrency from multiplying crash-loop respawns.

### Evidence and showcase

- 301 Python tests pass in the current tree.
- Combined statement+branch coverage is 90.50% with a two-decimal >=90.00% gate; statement coverage is 91.77% and branch-hit coverage is 86.33%.
- C++20 Release and ASan/UBSan tests pass.
- The wheel builds offline through the setuptools backend.
- A public-data showcase path acquires UCI HAR with provenance, runs model/shift/deployment experiments, and generates JSON + SVG + Markdown evidence.
- The offline smoke showcase proves the same experiment machinery without network access.

## Highest-value remaining weaknesses

### 1. Process isolation is not yet a complete OS security boundary

The backend child still runs under the service security principal. Environment clearing, resource limits and `no_new_privs` reduce ambient authority and blast radius, but a compromised native backend may still access filesystem/network resources available to that UID.

**Next step:** run the backend under a distinct identity and qualify namespace/seccomp/cgroup or service-manager restrictions with a deliberately malicious backend fixture.

### 2. Real ONNX Runtime native execution is not qualified

The narrow `onnxruntime-cpu` contract is implemented and tested against a controlled interface, but the optional native runtime is not installed in this qualification environment.

**Next step:** pin one ORT version and execute malformed/load/timeout/crash/memory/accuracy cases through the same isolated deployment and inference boundaries.

### 3. External anti-rollback needs real protected hardware state

The software transaction is well specified, but memory/file anchors cannot resist rollback of the storage containing them.

**Next step:** implement one TPM/secure-element adapter and perform provisioning, exhaustion, acknowledgement-loss, power-cut, and RMA tests.

### 4. HTTP application bounds do not replace ingress controls

ASGI controls apply after socket/TLS acceptance.

**Next step:** define and test Uvicorn/service-manager/reverse-proxy connection, keep-alive, backlog and slow-client limits.

### 5. Showcase real-data execution still needs retained network-enabled evidence

The UCI HAR acquisition/loader/experiment path is implemented and schema-tested locally, but this sandbox cannot download the 58 MB upstream archive.

**Next step:** run the manual GitHub `showcase-real-data` workflow and retain its generated evidence with a release/tag.

### 6. Target-device claims remain intentionally absent

Host timing is not Jetson/mobile/industrial timing.

**Next step:** qualify cold/warm latency, RSS/device memory, power, thermal throttling, restart behavior and soak on one explicitly named target device.

## Language decision

Continue with **Python + C++20**. Python is appropriate for orchestration, policy, cryptographic lifecycle integration, experiments, API and fault injection. C++20 remains appropriate for native kernels and production inference SDK integration. A third language is not justified until a small privileged supervisor or another independently deployable systems component has a concrete requirement for it.
