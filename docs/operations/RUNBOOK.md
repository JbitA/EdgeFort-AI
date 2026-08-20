# Operations Runbook

## Before staging a release

1. Verify the candidate semantic version and release sequence.
2. Validate the public signing key is trusted and not revoked.
3. Confirm the candidate runtime ABI is supported.
4. Run `make qualify` in the intended build environment.
5. Retain the qualification evidence, manifest and artifact hash.

## Candidate rejected

Inspect deployment state `last_error`. Common causes are accuracy regression, poor calibration, latency, memory, signature failure, ABI mismatch or an old release sequence. Do not bypass the health gate merely to complete a rollout.

## Rollback

Call the deployment library `rollback()` or the operator integration built around it. Rollback changes the active slot to the immediately previous version but does not reduce the anti-rollback release floor.

## Telemetry connectivity loss

Events should continue to append to the bounded local spool. Monitor `dropped`, `bytes`, `persistent_id_records` and `legacy_records`. When connectivity is restored, flush in bounded batches. Use `flush_records()` with the record `event_id` as the remote idempotency key when duplicate side effects matter. `flush()` remains suitable for sinks that do not support idempotency. Corrupt records are counted and discarded instead of blocking later valid data.

## Drift

Fit a `PSIProfile` on an accepted reference dataset. Monitor mean/max PSI and fraction of features above the local operational threshold. PSI indicates distribution shift; it does not by itself prove accuracy degradation. Escalate to model-specific validation before automatic rollback unless the application has a validated drift-to-risk policy.

## Integrity-event handling

If startup reports that deployment state is missing while slot data exists, do **not** delete the slots merely to make startup succeed. Treat this as an integrity/recovery event: preserve the deployment directory, determine why `state.json` disappeared, compare slot contents against the signed registry, and restore/reconstruct state only through an audited recovery procedure.

If active-slot verification fails, stop serving that runtime. Re-register/redeploy only from a verified signed registry source; do not copy model bytes directly into an active slot.

## Process ownership

Registry, deployment and telemetry mutation use POSIX advisory inter-process locks, so cooperating processes sharing one root serialize writes. Prefer one lifecycle agent per device anyway: advisory locks do not constrain hostile/root processes, and a real monotonic rollback anchor needs an explicit single-writer/provisioning policy. Multiple HTTP worker processes also multiply aggregate inference capacity unless admission is coordinated above them.

## External rollback anchor

The default application-only floor does not resist rollback of the whole filesystem. Devices with that threat must construct `ABDeployer` with a production `RollbackAnchor` backed by TPM/secure-element/boot-chain/other protected monotonic state. `MemoryRollbackAnchor` and `FileRollbackAnchor` are test doubles only.

If startup reports that the external anchor is ahead of local state without a valid recovery journal, preserve the deployment directory and anchor state and treat it as an integrity/recovery incident. Do not reset the hardware counter or edit `state.json` to make the values match.

## Hung inference during shutdown

Graceful executor shutdown uses a bounded wait. A timeout allows the service process to continue terminating because workers are daemon threads, but it does not cancel a running native call. Repeated shutdown timeouts are a backend-health incident and should trigger process/device watchdog policy, not a larger timeout by default.
