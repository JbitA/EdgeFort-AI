# ADR 0004: Bounded async admission and POSIX filesystem locking

- Status: Accepted
- Date: 2026-08-20

## Context

The reference service previously had a bounded queue primitive, but the HTTP endpoint invoked inference directly. Separately, registry, deployment, and telemetry mutations were thread-safe but not serialized across cooperating processes. Those properties were insufficient for a production-oriented edge daemon: overload control has to exist at the public coroutine boundary, and filesystem transactions must not race merely because two worker processes exist.

## Decision

1. `/v1/predict` submits work through a fixed-worker `BoundedInferenceExecutor` from an async endpoint.
2. Admission is bounded by both an outer HTTP in-flight limit and an internal waiting queue. The production default is reject-on-full.
3. Queue deadlines apply to **start of execution**. Running inference is never asynchronously killed from Python.
4. Registry mutation, deployment transitions, and telemetry spool mutation use POSIX advisory `flock` locks stored separately from atomically-renamed state files.
5. Deployment state reads remain lock-free because complete state files are committed with atomic rename. A shared process lock is acquired when a runtime snapshot must be read and authenticated consistently.

## Consequences

- HTTP overload is explicit: queue saturation returns 429 and queue-start expiry returns 504.
- Only fixed inference workers execute model code; accepted waiting work is bounded.
- Cooperative multi-process writers cannot interleave registry/deployment/spool transactions.
- A deployment benchmark can hold the lifecycle writer lock while an already-cached runtime continues serving.
- Advisory locks do not defend against root or another actor deliberately ignoring the locking protocol.
- Web-server socket/backlog limits, native backend thread pools, and fleet-wide admission are separate controls and remain deployment responsibilities.
- Shutdown cannot safely interrupt arbitrary native inference already running; production backends should provide bounded execution or process isolation if hard shutdown deadlines are required.

## Rejected alternatives

### Add another service language

Rewriting the device control plane in Go or Rust would increase the qualification surface without solving the immediate admission and persistence invariants. Python remains appropriate for the reference lifecycle/control plane; C++20 remains the native backend boundary.

### Use an unbounded web-server thread pool plus a queue inside inference

That permits requests to consume upstream worker/thread resources before bounded admission. Async submission makes the application-level admission point explicit.

### Lock `state.json` directly

The state file is replaced atomically. Locking that pathname would couple lock lifetime to inode replacement and is error-prone. A dedicated `.deployment.lock` file has stable identity.
