# Threat Model

## Assets

- approved model bytes and runtime compatibility metadata;
- signing trust roots and revocation state;
- release sequence / anti-rollback floor;
- active deployment state and crash-recovery metadata;
- inference integrity and availability;
- telemetry availability and, where applicable, integrity.

## Threats addressed

### Artifact substitution and verification/load TOCTOU

Mitigation: SHA-256 model digest inside the signed manifest; no-follow regular-file opens; metadata/model opens relative to a stable directory descriptor; hashing, NPZ preflight and model loading use the same model file handle. Registry, staged candidate, post-rename slot, rollback target and active runtime are re-authenticated.

### Alternate signed artifact/release identity ambiguity

Mitigation: executable slot manifests must equal the canonical signed manifest registered for that version. Semantic version is not treated as sufficient artifact identity, and one release sequence may belong to only one published semantic version in a registry. Registry publication fails closed if it cannot establish that uniqueness.

### Signed-manifest parser ambiguity

Mitigation: strict JSON parsing rejects duplicate keys/non-finite constants; manifest parsing rejects unknown/missing fields, boolean-as-number coercion and string/number coercion. Signature verification therefore authenticates one typed schema rather than a permissively reconstructed equivalent.

### Forged release

Mitigation: Ed25519 detached signatures and a trusted public-key keyring.

### Compromised/deprecated signing key

Mitigation: explicit key revocation; key-rotation tests verify new-key acceptance and old-key rejection. Private signing-key custody remains an external operational responsibility.

### Runtime-format confusion

Mitigation: supported `model_kind` values are bound to explicit signed runtime ABI identifiers. ABI mismatch is rejected before execution.

### Signed rollback attack

Mitigation: monotonic release sequence and persistent trusted floor. Startup fails closed rather than silently resetting state when populated slots exist without `state.json`. Recovery also rejects a referenced slot whose signed release sequence exceeds the durable floor.

When configured, an external `RollbackAnchor` binds the local floor to protected monotonic state. Deployment journals the exact target before advancing the external floor; startup can complete a crash between external advancement and local state commit only after authenticating the journal, canonical release and target slot. Unexplained anchor/state disagreement fails closed.

The included memory/file anchors are software test doubles. Whole-filesystem rollback resistance is claimed only after integrating a genuine external monotonic primitive and qualifying its power-loss/provisioning semantics on target hardware.

### Partial/crashed deployment

Mitigation: inactive-slot staging, candidate health gate, authenticated staged bytes, post-rename authentication, atomic state commit, backup restoration/discard based on durable state, and removal of uncommitted slot residue.

### Resource-hostile compressed model

Mitigation for the qualified NPZ reference formats: exact-member validation, bounded declared expanded size, bounded NPY header parsing, object-dtype rejection, and model-specific shape/dtype checks before NumPy allocation. The default expanded-size ceiling is 256 MiB and is configurable.

Future backends require their own preflight/resource contract.

### Resource-exhaustion inference input/output

Mitigation: protected prediction requests are authenticated before body parsing, admitted through a bounded pre-body in-flight counter, limited by strict HTTP framing plus a streaming request-byte ceiling, validated for tensor shape/finite values, and submitted to a fixed-worker executor with a bounded waiting queue. Saturation and queue-start expiry have explicit HTTP semantics. Backend outputs are independently bounded by rank, request batch, class count and total element count and must be numeric/finite before prediction or serialization.

Network/socket/TLS/backlog denial of service occurs before this application boundary and remains a server/OS deployment responsibility. ASGI framing checks do not prove proxy/server request-smuggling resistance.

### Hung, crashed or memory-runaway inference backend

Mitigation: optional process execution mode constructs the runtime inside one spawned child per executor worker from an authenticated generation-tagged artifact snapshot. A hard execution timeout terminates/restarts the child; child crashes are contained; readiness repairs an idle dead child. An optional POSIX `RLIMIT_AS` ceiling can be applied before model load.

Thread mode intentionally does not promise hard cancellation. `RLIMIT_AS` is target-specific and is not a substitute for cgroups/service-manager limits. Multiple process workers multiply model memory. Deployment-time candidate loading/benchmarking is still in the lifecycle process and remains a separate watchdog/isolation gap.

### Unauthenticated inference access

Mitigation: production FastAPI construction requires an API key and suppresses interactive API docs by default. Raw-header middleware rejects missing/duplicate/wrong keys before protected prediction-body allocation.

This is a reference boundary, not a substitute for mTLS, device identity, secure secret storage or rotation.

### Telemetry duplicate side effects after retry

Mitigation: each new spool record has a stable 128-bit event ID. Record-aware flush exposes that ID as a sink idempotency key, so a sink can deduplicate replay after acknowledgement loss. Legacy records remain readable but are explicitly identified as lacking persistent IDs. This does not make telemetry tamper evident.

### Cooperative multi-process filesystem races

Mitigation: registry mutation, deployment transitions and telemetry spool mutation use POSIX advisory `flock` locks. Sensitive lock/spool/model/state final components reject symlinks or require regular files.

Advisory locks do not protect against a hostile privileged process that ignores the locking protocol.

## Threats not fully addressed

- host/root compromise or arbitrary modification by a privileged local actor;
- signing-key theft or malicious authorized signer;
- secure boot / measured boot / OS update chain;
- rollback of the entire filesystem snapshot when no qualified external monotonic anchor is configured, or compromise/reset of that anchor itself;
- kernel, driver, firmware and accelerator vulnerabilities;
- network/OS-level connection floods, slow clients, TLS handshake exhaustion and proxy buffering;
- hard cancellation is implemented only for the optional reference process mode; deployment-time health benchmarking and real ONNX/TensorRT/vendor backends still require dedicated isolation/target qualification;
- device identity, mTLS and platform secret storage/rotation;
- target-device latency/WCET, thermal, power, flash endurance and power-loss qualification;
- signed trust-store/keyring update format and fleet approval workflow;
- tamper-evident telemetry audit semantics; idempotency identifiers are implemented but require sink-side deduplication;
- side-channel attacks;
- ONNX Runtime/TensorRT/other backend-specific parser and allocator risks until those backends are explicitly integrated and qualified.
