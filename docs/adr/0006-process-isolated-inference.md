# ADR 0006: Optional process-isolated inference with authenticated artifact bootstrap

## Status

Accepted for the reference implementation; production backends still require target-specific qualification.

## Context

A bounded request queue and queue-start deadline do not bound a backend after execution begins. Python cannot safely and generally terminate a thread that is inside NumPy, ONNX Runtime, TensorRT, a vendor SDK, or another native call. Allowing an indefinitely blocked worker undermines service availability and graceful-restart guarantees. Passing a live runtime/session object to another process is also a poor production contract because many native sessions are not safely pickleable and object serialization expands the trust boundary.

Backend output is another resource boundary: a malformed or compromised backend could return an unexpected rank, batch size, class count, non-finite values, or an extremely large tensor even when the input request is bounded.

## Decision

Keep thread execution as the lightweight default and add an opt-in `process` execution mode.

Each inference worker owns one spawned child process. The parent lifecycle/control plane:

1. validates the input tensor;
2. obtains a generation-consistent snapshot of the active release;
3. re-authenticates the exact active-slot model inode and canonical signed manifest;
4. reads only the signed bounded model bytes from that same open handle;
5. sends model bytes, signed `model_kind`/`runtime_abi`, generation and inference limits to the child;
6. applies a hard wall-clock timeout only to the actual inference operation;
7. terminates, then escalates to kill, a child that exceeds that deadline;
8. validates returned logits again in the parent before prediction/serialization.

The child reconstructs its own runtime from authenticated bytes. The normal `InferenceEngine` path therefore does not pickle a live runtime or native session. A runtime-object compatibility path remains only for custom/test engines.

Process mode may additionally configure `process_memory_limit_mb`. On the qualified POSIX/Linux path the child applies `RLIMIT_AS` before loading the model. This limit is disabled by default because appropriate virtual-address-space budgets are backend- and device-specific.

Queue-start timeout and execution timeout are separate operational events. The HTTP boundary reports them separately. `/readyz` verifies process workers and restarts an idle child that died outside a request.

## Subsequent hardening

The production-oriented reference profile now also clears the child environment before backend loading, disables core dumps, applies `no_new_privs` on Linux, caps open file descriptors, and uses an executor-wide rolling restart budget so worker count cannot multiply a crash-loop respawn allowance. IPC was migrated from Python object transport to a strict length-prefixed raw-byte protocol over `socketpair`, removing child-to-parent pickle deserialization from the isolation boundary.

## Consequences

Positive:

- a hung or crashed backend is contained to its child process;
- execution deadlines have a real termination mechanism instead of unsafe thread cancellation;
- lifecycle trust remains in the parent process;
- production backends can construct native sessions inside the process that owns them;
- model bytes and backend outputs remain bounded at the IPC boundary;
- optional address-space limits reduce the blast radius of runaway allocations;
- metrics distinguish queue overload, execution timeout, child crash and restart.

Costs and limitations:

- every process worker owns a separate runtime and therefore multiplies model memory;
- the current reference bootstrap copies model bytes through a bounded versioned socketpair protocol, which is unsuitable for very large models and should evolve to an authenticated file-descriptor/shared-memory/backend bootstrap contract;
- process spawn and model load use a separate control timeout and are not included in the inference execution deadline;
- `RLIMIT_AS` can interact poorly with runtimes that reserve large virtual address ranges and must be target-qualified;
- cgroups/service-manager limits remain preferable for whole-service memory/CPU containment;
- deployment-time candidate parsing and benchmarking now have an optional isolated qualification process; real ONNX/TensorRT/vendor backends still require native qualification;
- the default thread mode intentionally remains non-killable once execution starts.

## Rejected alternatives

**Asynchronously kill Python worker threads.** Rejected because Python and native libraries do not provide a safe, general thread-termination mechanism.

**Fork per request.** Rejected because fork semantics around multithreaded/native runtimes are unsafe and per-request startup cost is unnecessary.

**Pickle the production runtime/session into the child.** Rejected because it couples process isolation to Python object serialization and does not generalize safely to native sessions.

**Make process mode mandatory immediately.** Rejected because the reference thread path remains useful and real accelerator backends need target-specific memory/startup qualification before process isolation can be the universal default.
