# Architecture

## Design goals

The platform separates five concerns that are frequently coupled in edge-AI demos:

1. **artifact trust** — immutable bytes, manifest, signature and key trust;
2. **runtime compatibility** — model kind and runtime ABI are explicit;
3. **release policy** — health gates and anti-rollback sequence;
4. **operational state** — active/previous slot and crash-consistent state file;
5. **inference operations** — bounded admission/execution, metrics, offline telemetry and drift.

The local registry is intentionally filesystem-backed. This keeps lifecycle semantics directly testable and avoids making correctness depend on an external cloud control plane.

## Trust and load flow

A release is not trusted merely because it exists in a directory:

```text
artifact bytes -> SHA-256
manifest ------> Ed25519 detached signature
public key ----> trusted / revoked keyring
all checks pass -> immutable canonical registry entry
                         |
                         v
                 stable no-follow file handle
                         |
                 NPZ structural preflight
                         |
                 model allocation / runtime load
```

Registry metadata and model bytes are opened relative to a stable directory descriptor. The model hash is computed through the same file handle later passed to the loader, closing the path-reopen verification/load TOCTOU gap. Deployment re-authenticates staged candidates, post-rename slots, rollback targets and active runtime snapshots. A slot manifest must also equal the canonical registry manifest for its semantic version. Published release sequences are unique within a registry, so a monotonic anchor value has one canonical signed release identity. Signed manifest JSON is parsed with duplicate-key rejection and strict non-coercive field types before signature verification.

## Model preflight

The qualified NPZ reference formats are inspected before NumPy allocation. Preflight enforces:

- exact expected members and no duplicates/nested paths;
- allowed compression methods and no encryption;
- a configurable total declared expanded-byte ceiling;
- bounded NPY headers and supported NPY versions;
- no object dtypes;
- rank, shape, dtype and cross-array model-schema constraints.

This is deliberately backend-specific. A future ONNX/TensorRT loader must define equivalent pre-allocation/resource invariants rather than bypass this boundary.

## Deployment state and recovery

`state.json` contains generation, active slot, previous slot, slot-to-version mapping, highest trusted release sequence and status. State writes use temporary-file + `fsync` + atomic rename. Slot replacement is staged separately.

The deployer serializes transitions across threads and cooperating POSIX processes. Recovery authenticates `.backup-*` directories against durable state, restores pre-commit backups, discards stale post-commit backups, removes an uncommitted first-write slot, and rejects referenced slot releases above the durable anti-rollback floor. The active slot is never modified in place during candidate benchmarking.

## Anti-rollback invariant

Let `R_max` be `highest_release_sequence`. Normal deployment requires:

```text
candidate.release_sequence > R_max
```

Operator rollback changes the active slot but does not reduce `R_max`. This permits recovery to a previously installed artifact while preventing later replay of older signed artifacts as new updates.

By default, `R_max` is application state and therefore cannot resist rollback of the entire filesystem. When an external `RollbackAnchor` is configured, the deployer requires local and external floors to agree and commits a new release using a two-persistence-domain transaction:

```text
verified target slot
      |
write/fsync pending journal
      |
advance external monotonic floor
      |
atomic state.json commit
      |
remove journal / obsolete backup
```

If power fails after the external advance but before local state commit, startup authenticates the journal, canonical registry release and target slot before completing the state transition. Missing/conflicting evidence fails closed. `MemoryRollbackAnchor` and `FileRollbackAnchor` exist to qualify this software protocol; only a separately qualified TPM/secure-element/boot-chain adapter can establish whole-filesystem rollback resistance.

## Runtime lifecycle

The `InferenceEngine` caches an active runtime by `(deployment generation, semantic version)`, not version alone. On cache refresh it obtains `(runtime, version, generation)` from one shared-lock deployment snapshot, preventing a state/runtime mismatch during concurrent promotion or rollback.

Runtime output is treated as an untrusted backend boundary. Before prediction or API serialization the engine requires a numeric finite rank-2 tensor, exact request-batch correspondence, bounded class count and bounded total output elements. Process workers perform an independent check before IPC serialization and the parent validates again.

### Optional process-isolated execution

The library default remains lightweight thread mode, while `configs/reference.yaml` selects process mode as the production-oriented reference profile. In process mode each executor worker owns one `spawn`-created backend child. The parent does not pickle the production runtime/session. It re-authenticates the exact active-slot inode, captures only the signed bounded model bytes plus `model_kind`/`runtime_abi`, and sends that generation-tagged snapshot to the child. The child loads its own runtime. A hard execution deadline terminates/restarts the child; an unexpected child exit is contained. `/readyz` also repairs an idle dead child.


Before backend/model loading, disposable process workers apply the configured sandbox controls: inherited environment removal, core-dump suppression, a file-descriptor ceiling, optional address-space limit, and Linux `no_new_privs`. An executor-wide rolling restart budget limits crash-loop amplification across all workers. These are containment controls, not a substitute for a distinct OS security principal, namespaces/seccomp/cgroups, or a separately supervised backend service.

An optional `process_memory_limit_mb` applies POSIX `RLIMIT_AS` before child model load. It is intentionally disabled by default because address-space requirements are backend/device-specific. Whole-service cgroups/service-manager limits remain a deployment concern. The current safe IPC uses a bounded versioned socketpair protocol. For large future models, copying full model bytes should be replaced by an authenticated file-descriptor or sealed shared-memory/backend bootstrap contract.

## HTTP admission flow

```text
socket/server acceptance  [external server/OS boundary]
        |
        v
raw-header API-key check
        |
        v
bounded prediction in-flight reservation
        |
        v
streaming request-byte limit
        |
        v
Pydantic + tensor validation
        |
        v
bounded async waiting queue
        |
        v
fixed inference worker(s)
        |
        +--> thread mode: active runtime
        |
        `--> process mode: authenticated artifact snapshot
                         -> spawned backend child
                         -> hard execution deadline / restart
```

The production default is reject-on-full. Queue saturation maps to HTTP 429 with `Retry-After`; queue-start expiry maps to 504. Ambiguous HTTP body framing (duplicate `Content-Length`, `Content-Length` plus `Transfer-Encoding`, malformed lengths, or declared/observed length disagreement) is rejected at the ASGI boundary. In thread mode, Python does not asynchronously kill a running backend call. In process mode, execution timeout terminates/restarts the child and is reported separately from queue timeout. Graceful shutdown remains bounded in both modes.

## Concurrency and locking

- Registry mutation: exclusive POSIX advisory lock.
- Deployment transition/recovery: exclusive POSIX advisory lock.
- Active runtime snapshot: shared deployment lock.
- Deployment state reads: lock-free atomic-file reads so cached inference can continue while a candidate is benchmarked.
- Telemetry spool mutation: exclusive POSIX advisory lock; stats use shared locking. New records carry persistent event IDs for retry idempotency.
- Engine cache and metrics: process-local thread locks.

These locks protect **cooperating processes**, not hostile root or actors that deliberately ignore the protocol.


## Showcase/evidence flow

The GitHub-facing public-data experiment uses a separate acquisition boundary rather than committing raw data. `edgeai dataset fetch uci-har` streams the official UCI archive through a byte ceiling, rejects unsafe ZIP structure, extracts only required files, and records source URL/DOI/license plus the exact observed archive SHA-256 in `provenance.json`.

`edgeai showcase` then trains/quantizes the compact reference model, records quality/footprint trade-offs, fits PSI drift evidence, and drives the real signed-registry/A-B-deployment path including a validly signed but intentionally degraded candidate. One JSON result is rendered into a GitHub-readable Markdown report and SVG summary. The experiment therefore exercises the same lifecycle primitives presented in the architecture rather than being a disconnected notebook.

## Extension model

Python remains the lifecycle/control/reference plane. C++20 remains the native kernel/backend integration boundary. A new backend should add an explicit `model_kind`, signed `runtime_abi`, parser/preflight contract, resource bounds, health-gate coverage, and target qualification. Adding another production language is not justified until a concrete systems requirement outweighs the additional trusted and supply-chain surface.
