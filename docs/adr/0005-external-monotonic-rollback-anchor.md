# ADR 0005: External monotonic anti-rollback anchor and recoverable commit journal

- Status: accepted
- Date: 2026-08-20

## Context

The application anti-rollback floor in `state.json` prevents normal replay of older signed releases, but it cannot detect rollback of the entire deployment filesystem. Moving the floor to a TPM, secure element, protected boot counter, or remote trusted monotonic service creates a second persistence domain. Updating two independent persistence domains cannot be made atomically with a filesystem rename.

The dangerous ordering choices are:

1. **state first, anchor second** — a crash or snapshot rollback before the anchor update can temporarily lose the anti-rollback guarantee;
2. **anchor first, state second** — anti-rollback remains fail-closed, but a crash between the two writes can make the local state appear stale and strand availability unless recovery knows the intended transition.

## Decision

Use an optional `RollbackAnchor` protocol with `read_floor()` and durable `advance_to(release_sequence)` operations. Production adapters must place the protected monotonic value outside the rollbackable deployment filesystem.

When an external anchor is configured, deployment uses this order after candidate health qualification and post-rename verification:

```text
verified target slot
      |
      v
write + fsync pending anchor journal
      |
      v
advance external monotonic anchor   <-- irreversible trust step
      |
      v
atomically commit state.json
      |
      v
remove journal and obsolete backup
```

The journal records the previous generation/floor and the exact target slot/version/release sequence. Release sequences are required to be globally unique in the immutable registry so recovery cannot map one anchor value to multiple signed releases. The same exact integer domain (`0..2^63-1`, with `-1` reserved for an unprovisioned/empty floor) is enforced by manifests, deployment state, recovery journals, and anchor adapters.

If `advance_to()` raises after the hardware may already have committed the new value, the live deployer enters a **recovery-required poison state** and refuses further state-dependent service. It does not assume the anchor stayed unchanged. A fresh deployer/recovery pass re-reads the anchor and resolves the journal. Cooperating long-lived peers refresh their cached anchor when durable deployment state changes or a pending anchor journal appears, avoiding a hardware read on every inference.

On restart:

- local floor == anchor floor, no committed transition: restore the durable local state and discard an abandoned pre-anchor journal;
- local floor == anchor floor, state already committed: authenticate the target and remove stale journal/backup residue;
- anchor floor > local floor: require an exact recovery journal, a uniquely matching canonical registry release, and an authenticated target slot, then complete the local state commit;
- local floor > anchor floor, missing journal, conflicting journal, invalid target, or ambiguous release-sequence ownership: fail closed.

`MemoryRollbackAnchor` and `FileRollbackAnchor` are software test doubles. `FileRollbackAnchor` is explicitly **not** evidence against whole-filesystem rollback.

## Consequences

### Positive

- whole-filesystem rollback can be detected when a genuine external monotonic adapter is used;
- anchor-first ordering preserves the security invariant across crash/power loss;
- the recovery journal avoids unnecessary device bricking for the normal crash window between anchor and state commit;
- the adapter contract can be exercised in CI before target TPM/secure-element integration.

### Negative / residual

- a production hardware adapter is still target-specific and externally qualified;
- a privileged attacker who can compromise the anchor itself is outside this control;
- cooperative peer deployments are observed without a hot-path anchor read, but an anchor-only change performed outside the deployment protocol still relies on lifecycle-agent ownership or a qualified refresh/watch policy;
- hard guarantees require validating real device power-loss durability and counter semantics, not only the software test doubles.

## Adapter requirements

A production adapter must document and qualify:

- monotonicity across reboot and power loss;
- authenticated access/authorization policy;
- durable completion semantics for `advance_to`;
- supported numeric range and exhaustion behavior;
- behavior when hardware is unavailable or reports corruption;
- provisioning, replacement, RMA and recovery procedures;
- concurrency/single-writer assumptions.
