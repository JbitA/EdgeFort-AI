# System Requirements

| ID | Requirement |
|---|---|
| REQ-001 | Registry shall reject modified model bytes. |
| REQ-002 | Registry shall reject unknown or revoked signing keys. |
| REQ-003 | Registry versions shall be immutable. |
| REQ-004 | Normal deployment shall reject a release sequence at or below the trusted anti-rollback floor. |
| REQ-005 | Candidate activation shall be health-gated before becoming active. |
| REQ-006 | Failed candidate validation shall not replace the active model. |
| REQ-007 | Deployment state shall be atomically persisted. |
| REQ-008 | Explicit rollback shall preserve the highest trusted release sequence. |
| REQ-009 | Inference shall reject non-finite and structurally invalid input. |
| REQ-010 | Public inference shall enforce batch/resource limits. |
| REQ-011 | The inference queue shall have explicit overload behavior. |
| REQ-012 | Offline telemetry shall survive process interruption after append and remain bounded. |
| REQ-013 | Corrupt telemetry records shall not prevent valid records from being retried. |
| REQ-014 | Runtime metrics shall be available without an external monitoring service. |
| REQ-015 | Production API construction shall fail without authentication material. |
| REQ-016 | Quantized artifacts shall retain bounded accuracy loss on the reference workload. |
| REQ-017 | Native int8 primitives shall be tested in Release and sanitizer builds. |
| REQ-018 | Package branch-aware coverage shall remain at least 90%. |
| REQ-019 | Input-distribution drift shall be measurable against a fitted reference profile. |
| REQ-020 | Release evidence shall distinguish validated software from hardware/integration claims not tested here. |
| REQ-021 | Executable slot bytes shall be cryptographically re-verified before runtime load. |
| REQ-022 | A supported model kind shall execute only with its signed expected runtime ABI. |
| REQ-023 | Deployment-state parsing shall reject unknown fields, invalid slots, invalid versions and inconsistent active/previous state. |
| REQ-024 | The HTTP service shall bound request bytes before schema/model allocation. |
| REQ-025 | When configured, an external monotonic rollback anchor shall fail closed on unexplained disagreement with local deployment state. |
| REQ-026 | External-anchor deployment shall be recoverable across a crash after anchor advancement but before local state commit. |
| REQ-027 | Published release sequences shall be unique across semantic versions in one registry. |
| REQ-028 | Signed manifest parsing shall reject unknown/duplicate fields and type coercion. |
| REQ-029 | Runtime configuration shall reject duplicate/unknown keys and ambiguous boolean/numeric coercion. |
| REQ-030 | New telemetry spool records shall retain a retry-stable event identifier and expose it to idempotent sinks. |
| REQ-031 | Graceful inference-executor shutdown shall support a bounded wait without unsafe thread cancellation. |
| REQ-032 | Inference shall reject malformed, non-finite, batch-mismatched or configured-limit-exceeding backend output before prediction/serialization. |
| REQ-033 | Optional process execution shall bound backend execution time by terminating/restarting a child without unsafe thread cancellation. |
| REQ-034 | Process-mode production bootstrap shall construct the child runtime from an authenticated generation-tagged artifact snapshot rather than require pickling a live runtime session. |
| REQ-035 | The ASGI boundary shall reject ambiguous or inconsistent request-body framing before prediction parsing. |
| REQ-036 | Optional process memory containment shall fail closed during child startup when the configured POSIX address-space ceiling cannot be applied. |
| REQ-037 | Committed SBOM/qualification constraints shall be regenerable from source metadata, and CI shall detect non-reproducible wheel bytes for a fixed build epoch. |
