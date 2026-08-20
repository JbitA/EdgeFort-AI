from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import io
import math
import multiprocessing as mp
import os
import resource
import socket
import threading
import time

import numpy as np

from .errors import (
    BackendProcessError,
    DeadlineExceeded,
    DeploymentError,
    ExecutionDeadlineExceeded,
    ExecutorClosed,
    QueueDropped,
    QueueFull,
    ValidationError,
)
from .sandbox import apply_process_sandbox
from .ipc import (
    FRAME_PREFIX_BYTES,
    IPCProtocolError,
    MAX_HEADER_BYTES,
    decode_error,
    decode_frame,
    decode_generation,
    decode_load_artifact,
    decode_predict,
    decode_ready,
    decode_result,
    decode_simple,
    decode_startup_error,
    encode_error,
    encode_generation,
    encode_load_artifact,
    encode_predict,
    encode_ready,
    encode_result,
    encode_simple,
    encode_startup_error,
    frame_length_prefix,
    parse_frame_length,
)


class OverflowPolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    REJECT = "reject"


class ExecutionMode(str, Enum):
    THREAD = "thread"
    PROCESS = "process"


@dataclass(frozen=True)
class WorkItem:
    payload: object
    enqueued_monotonic: float
    deadline_monotonic: float | None = None


class BoundedInferenceQueue:
    def __init__(self, capacity=32, policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.policy = OverflowPolicy(policy)
        self._q = deque()
        self._cv = threading.Condition(threading.Lock())
        self.dropped = 0
        self.expired = 0
        self.accepted = 0

    @staticmethod
    def _deadline(now: float, deadline_s: float | None) -> float | None:
        if deadline_s is not None and (not math.isfinite(deadline_s) or deadline_s < 0):
            raise ValueError("deadline_s must be finite and >= 0")
        return None if deadline_s is None else now + deadline_s

    def push_with_evicted(self, payload, deadline_s: float | None = None):
        now = time.monotonic()
        item = WorkItem(payload, now, self._deadline(now, deadline_s))
        evicted = None
        with self._cv:
            if len(self._q) >= self.capacity:
                if self.policy == OverflowPolicy.REJECT:
                    raise QueueFull("inference queue full")
                if self.policy == OverflowPolicy.DROP_NEWEST:
                    self.dropped += 1
                    return False, None
                evicted = self._q.popleft()
                self.dropped += 1
            self._q.append(item)
            self.accepted += 1
            self._cv.notify()
            return True, evicted

    def push(self, payload, deadline_s: float | None = None):
        accepted, _ = self.push_with_evicted(payload, deadline_s=deadline_s)
        return accepted

    def pop_with_expired(self, wait_s: float = 0.0):
        if not math.isfinite(wait_s) or wait_s < 0:
            raise ValueError("wait_s must be finite and >= 0")
        end = time.monotonic() + wait_s
        expired_items = []
        with self._cv:
            while True:
                now = time.monotonic()
                while self._q:
                    item = self._q.popleft()
                    if item.deadline_monotonic is not None and item.deadline_monotonic <= now:
                        self.expired += 1
                        expired_items.append(item)
                        continue
                    return item, expired_items
                remaining = end - now
                if remaining <= 0:
                    return None, expired_items
                self._cv.wait(remaining)

    def pop(self):
        item, _ = self.pop_with_expired(0.0)
        return item

    def expire_payload(self, payload) -> bool:
        """Remove one still-queued payload by identity and count it as expired."""
        with self._cv:
            for index, item in enumerate(self._q):
                if item.payload is payload:
                    del self._q[index]
                    self.expired += 1
                    return True
            return False

    def drain(self):
        with self._cv:
            items = list(self._q)
            self._q.clear()
            return items

    def wake_all(self):
        with self._cv:
            self._cv.notify_all()

    def stats(self):
        with self._cv:
            return {
                "depth": len(self._q),
                "capacity": self.capacity,
                "accepted": self.accepted,
                "dropped": self.dropped,
                "expired": self.expired,
            }


@dataclass
class _InferenceTask:
    payload: object
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    result: object | None = None
    error: BaseException | None = None
    notify: object | None = None

    def begin(self) -> bool:
        with self.lock:
            if self.done.is_set():
                return False
            self.started = True
            return True

    def complete(self, *, result=None, error: BaseException | None = None) -> None:
        callback = None
        with self.lock:
            if self.done.is_set():
                return
            self.result = result
            self.error = error
            self.done.set()
            callback = self.notify
        if callback is not None:
            callback()

    def expire_if_not_started(self) -> bool:
        callback = None
        with self.lock:
            if self.done.is_set() or self.started:
                return False
            self.error = DeadlineExceeded("inference request expired before execution")
            self.done.set()
            callback = self.notify
        if callback is not None:
            callback()
        return True

    def cancel_if_not_started(self) -> bool:
        with self.lock:
            if self.done.is_set() or self.started:
                return False
            self.done.set()
            return True


def _worker_validate_logits(logits, *, batch_size: int, limits: dict[str, int]):
    """Bound process-to-parent IPC before serializing backend output."""
    try:
        raw = np.asarray(logits)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValidationError("runtime output must be a numeric rank-2 array") from e
    if raw.ndim != 2:
        raise ValidationError("runtime output must be rank-2")
    if raw.shape[0] != batch_size:
        raise ValidationError("runtime output batch dimension mismatch")
    classes = raw.shape[1]
    if classes < 1 or classes > limits["max_output_classes"]:
        raise ValidationError("runtime output class count outside configured bounds")
    if raw.size > limits["max_output_values"]:
        raise ValidationError("runtime output element count outside configured bounds")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValidationError("runtime output must be numeric")
    try:
        normalized = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValidationError("runtime output cannot be represented as float32") from e
    if not np.isfinite(normalized).all():
        raise ValidationError("runtime produced non-finite output")
    return normalized


def _apply_process_memory_limit(memory_limit_bytes: int | None) -> None:
    """Compatibility helper retained for direct unit tests and library callers."""
    if memory_limit_bytes is None:
        return
    if (
        isinstance(memory_limit_bytes, bool)
        or not isinstance(memory_limit_bytes, int)
        or memory_limit_bytes <= 0
    ):
        raise ValueError("process memory limit must be a positive integer")
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


def _apply_process_security(memory_limit_bytes: int | None, *, nofile_limit: int) -> None:
    """Harden the disposable backend before any model/runtime parser is imported."""
    # Keep the address-space operation as a distinct compatibility seam: tests and
    # platform adapters can fail it independently, while the sandbox helper applies the
    # remaining ambient-authority reductions.
    _apply_process_memory_limit(memory_limit_bytes)
    apply_process_sandbox(
        memory_limit_bytes=None,
        nofile_limit=nofile_limit,
        clear_environment=True,
        no_new_privs=True,
    )


def _socket_remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("IPC operation deadline exceeded")
    return remaining


def _socket_recv_exact(sock: socket.socket, size: int, *, deadline: float | None) -> bytes:
    if size < 0:
        raise IPCProtocolError("invalid IPC receive size")
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        sock.settimeout(_socket_remaining(deadline))
        count = sock.recv_into(view[offset:])
        if count == 0:
            raise EOFError("isolated inference socket closed")
        offset += count
    return bytes(data)


def _socket_recv_frame(
    sock: socket.socket,
    *,
    max_frame_bytes: int,
    deadline: float | None,
) -> bytes:
    prefix = _socket_recv_exact(sock, FRAME_PREFIX_BYTES, deadline=deadline)
    frame_size = parse_frame_length(prefix)
    if frame_size < 2 or frame_size > max_frame_bytes:
        raise IPCProtocolError("IPC frame exceeds configured bound")
    return _socket_recv_exact(sock, frame_size, deadline=deadline)


def _socket_send_frame(sock: socket.socket, frame: bytes, *, deadline: float | None) -> None:
    if not isinstance(frame, bytes):
        raise IPCProtocolError("IPC frame must be bytes")
    data = frame_length_prefix(len(frame)) + frame
    view = memoryview(data)
    offset = 0
    while offset < len(data):
        sock.settimeout(_socket_remaining(deadline))
        count = sock.send(view[offset:])
        if count == 0:
            raise EOFError("isolated inference socket closed")
        offset += count


def _isolated_backend_main(
    channel: socket.socket,
    memory_limit_bytes: int | None = None,
    ipc_limits: dict[str, int] | None = None,
    send_timeout_s: float = 5.0,
    nofile_limit: int = 64,
) -> None:
    """Spawned worker using a strict raw-byte protocol; no runtime-message pickle."""
    runtime = None
    generation = None
    limits = dict(ipc_limits or {})
    required = {
        "max_batch",
        "max_input_features",
        "max_input_values",
        "max_output_classes",
        "max_output_values",
        "max_model_transfer_bytes",
    }
    try:
        if set(limits) != required or any(
            isinstance(limits[name], bool) or not isinstance(limits[name], int) or limits[name] < 1
            for name in required
        ):
            return
        max_incoming = MAX_HEADER_BYTES + 1 + max(
            limits["max_model_transfer_bytes"], limits["max_input_values"] * 4
        )
        send_timeout_s = float(send_timeout_s)
        try:
            _apply_process_security(memory_limit_bytes, nofile_limit=nofile_limit)
        except Exception as e:
            _socket_send_frame(
                channel,
                encode_startup_error(type(e).__name__, str(e)[:4096] or "resource setup failed"),
                deadline=time.monotonic() + max(send_timeout_s, 0.001),
            )
            return
        _socket_send_frame(
            channel,
            encode_ready(os.getpid()),
            deadline=time.monotonic() + max(send_timeout_s, 0.001),
        )
        while True:
            try:
                frame = _socket_recv_frame(channel, max_frame_bytes=max_incoming, deadline=None)
                header, _ = decode_frame(
                    frame,
                    max_payload_bytes=max(
                        limits["max_model_transfer_bytes"], limits["max_input_values"] * 4
                    ),
                )
            except (EOFError, OSError, IPCProtocolError):
                return
            op = header["op"]
            if op == "close":
                try:
                    decode_simple(frame, "close")
                except IPCProtocolError:
                    return
                return
            if op == "load_artifact":
                try:
                    (
                        requested_generation,
                        model_bytes,
                        model_kind,
                        runtime_abi,
                        max_uncompressed_bytes,
                    ) = decode_load_artifact(
                        frame,
                        max_model_bytes=limits["max_model_transfer_bytes"],
                    )
                    from .health import load_runtime

                    loaded = load_runtime(
                        io.BytesIO(model_bytes),
                        model_kind,
                        runtime_abi,
                        max_uncompressed_bytes=max_uncompressed_bytes,
                    )
                except Exception as e:
                    runtime = None
                    generation = None
                    try:
                        _socket_send_frame(
                            channel,
                            encode_error(
                                "load_error",
                                request_id=None,
                                generation=header.get("generation")
                                if isinstance(header.get("generation"), int)
                                and not isinstance(header.get("generation"), bool)
                                and header.get("generation") >= 0
                                else 0,
                                code=type(e).__name__,
                                message=str(e)[:4096] or "model load rejected",
                            ),
                            deadline=time.monotonic() + send_timeout_s,
                        )
                    except Exception:
                        return
                    continue
                runtime = loaded
                generation = requested_generation
                try:
                    _socket_send_frame(
                        channel,
                        encode_generation("loaded", generation),
                        deadline=time.monotonic() + send_timeout_s,
                    )
                except Exception:
                    return
                continue
            if op != "predict":
                return
            try:
                request_id, expected_generation, inputs = decode_predict(
                    frame,
                    max_batch=limits["max_batch"],
                    max_input_features=limits["max_input_features"],
                    max_input_values=limits["max_input_values"],
                )
            except IPCProtocolError:
                return
            if runtime is None or generation != expected_generation:
                try:
                    _socket_send_frame(
                        channel,
                        encode_error(
                            "error",
                            request_id=request_id,
                            generation=None,
                            code="BackendProcessError",
                            message="worker runtime generation mismatch",
                        ),
                        deadline=time.monotonic() + send_timeout_s,
                    )
                except Exception:
                    return
                continue
            try:
                logits = runtime.logits(inputs)
                logits = _worker_validate_logits(
                    logits,
                    batch_size=len(inputs),
                    limits=limits,
                )
                response = encode_result(request_id=request_id, logits=logits)
            except Exception as e:
                response = encode_error(
                    "error",
                    request_id=request_id,
                    generation=None,
                    code="ValidationError" if isinstance(e, ValidationError) else type(e).__name__,
                    message=str(e)[:4096] or "backend failure",
                )
            try:
                _socket_send_frame(
                    channel,
                    response,
                    deadline=time.monotonic() + send_timeout_s,
                )
            except Exception:
                return
    finally:
        try:
            channel.close()
        except Exception:
            pass


class _RollingRestartBudget:
    """Thread-safe executor-wide rolling budget for child-process respawns."""

    def __init__(self, *, limit: int, window_s: float, on_throttled):
        self.limit = limit
        self.window_s = window_s
        self._on_throttled = on_throttled
        self._times = deque()
        self._lock = threading.Lock()

    def reserve(self) -> None:
        now = time.monotonic()
        cutoff = now - self.window_s
        with self._lock:
            while self._times and self._times[0] <= cutoff:
                self._times.popleft()
            if len(self._times) >= self.limit:
                self._on_throttled()
                raise BackendProcessError("isolated inference worker restart budget exhausted")
            self._times.append(now)

    def current(self) -> int:
        now = time.monotonic()
        cutoff = now - self.window_s
        with self._lock:
            while self._times and self._times[0] <= cutoff:
                self._times.popleft()
            return len(self._times)


class _ProcessInferenceRunner:
    """Own one killable spawned backend process over strict non-pickle IPC."""

    def __init__(
        self,
        *,
        control_timeout_s: float,
        memory_limit_bytes: int | None,
        nofile_limit: int = 64,
        restart_limit: int = 8,
        restart_window_s: float = 60.0,
        restart_budget: _RollingRestartBudget | None = None,
        ipc_limits: dict[str, int],
        counters,
    ):
        self.control_timeout_s = control_timeout_s
        self.memory_limit_bytes = memory_limit_bytes
        self.nofile_limit = nofile_limit
        self.restart_limit = restart_limit
        self.restart_window_s = restart_window_s
        self.ipc_limits = dict(ipc_limits)
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._channel: socket.socket | None = None
        self._generation = None
        self._ever_started = False
        self._request_id = 0
        self._lock = threading.Lock()
        self._counters = counters
        self._restart_budget = restart_budget or _RollingRestartBudget(
            limit=restart_limit,
            window_s=restart_window_s,
            on_throttled=lambda: self._increment("process_restart_throttled"),
        )
        self._max_receive_frame_bytes = (
            MAX_HEADER_BYTES + 1 + self.ipc_limits["max_output_values"] * 4
        )

    def _increment(self, name: str) -> None:
        self._counters(name)

    def _reserve_restart(self) -> None:
        self._restart_budget.reserve()

    def start(self) -> None:
        stale_process = None
        stale_channel = None
        with self._lock:
            process = self._process
            if process is not None and process.is_alive() and self._channel is not None:
                return
            stale_process, stale_channel = self._process, self._channel
            self._process = None
            self._channel = None
            self._generation = None
        if stale_process is not None and self._ever_started and not stale_process.is_alive():
            self._increment("process_crashes")
        if stale_channel is not None:
            try:
                stale_channel.close()
            except OSError:
                pass
        if stale_process is not None:
            try:
                stale_process.join(0)
                stale_process.close()
            except (ValueError, OSError):
                pass
        with self._lock:
            # Another caller may have repaired the worker while stale resources were
            # being reaped. Do not create a second child in that case.
            if self._process is not None and self._process.is_alive() and self._channel is not None:
                return
            if self._ever_started:
                self._reserve_restart()
            parent, child = socket.socketpair()
            process = self._ctx.Process(
                target=_isolated_backend_main,
                args=(
                    child,
                    self.memory_limit_bytes,
                    self.ipc_limits,
                    self.control_timeout_s,
                    self.nofile_limit,
                ),
                daemon=True,
            )
            try:
                process.start()
            except Exception:
                parent.close()
                child.close()
                raise
            child.close()
            self._process = process
            self._channel = parent
            self._generation = None
            was_restart = self._ever_started
            self._ever_started = True
            self._increment("process_starts")
            if was_restart:
                self._increment("process_restarts")
        frame = self._receive_frame(
            deadline=time.monotonic() + self.control_timeout_s,
            deadline_error=False,
        )
        try:
            header, _ = decode_frame(frame, max_payload_bytes=0)
            if header["op"] == "startup_error":
                code, message = decode_startup_error(frame)
                self.abort()
                self._increment("process_crashes")
                raise BackendProcessError(
                    f"isolated inference worker failed resource setup ({code}): {message}"
                )
            decode_ready(frame)
        except BackendProcessError:
            raise
        except (IPCProtocolError, ValueError) as e:
            self.abort()
            self._increment("process_crashes")
            raise BackendProcessError("isolated inference worker failed startup handshake") from e

    def _detach(self):
        with self._lock:
            process, channel = self._process, self._channel
            self._process = None
            self._channel = None
            self._generation = None
        return process, channel

    def abort(self) -> None:
        process, channel = self._detach()
        if channel is not None:
            try:
                channel.close()
            except OSError:
                pass
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(0.25)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(0.25)
            finally:
                try:
                    process.close()
                except (ValueError, OSError):
                    pass

    def _send_frame(self, frame: bytes, *, deadline: float, deadline_error: bool) -> None:
        channel = self._channel
        process = self._process
        if channel is None or process is None or not process.is_alive():
            self.abort()
            self._increment("process_crashes")
            raise BackendProcessError("isolated inference worker is unavailable")
        try:
            _socket_send_frame(channel, frame, deadline=deadline)
        except (TimeoutError, socket.timeout) as e:
            self.abort()
            if deadline_error:
                self._increment("execution_timeouts")
                raise ExecutionDeadlineExceeded("inference execution deadline exceeded") from e
            self._increment("process_crashes")
            raise BackendProcessError("isolated inference worker control operation timed out") from e
        except (BrokenPipeError, EOFError, OSError, IPCProtocolError) as e:
            self.abort()
            self._increment("process_crashes")
            raise BackendProcessError("isolated inference worker transport failed") from e

    def _receive_frame(self, *, deadline: float, deadline_error: bool) -> bytes:
        channel = self._channel
        process = self._process
        if channel is None or process is None:
            raise BackendProcessError("isolated inference worker is unavailable")
        try:
            return _socket_recv_frame(
                channel,
                max_frame_bytes=self._max_receive_frame_bytes,
                deadline=deadline,
            )
        except (TimeoutError, socket.timeout) as e:
            self.abort()
            if deadline_error:
                self._increment("execution_timeouts")
                raise ExecutionDeadlineExceeded("inference execution deadline exceeded") from e
            self._increment("process_crashes")
            raise BackendProcessError("isolated inference worker control operation timed out") from e
        except (EOFError, OSError, IPCProtocolError) as e:
            self.abort()
            self._increment("process_crashes")
            raise BackendProcessError("isolated inference worker exited or violated IPC protocol") from e

    def load(self, prepared) -> None:
        """Ensure the child owns the exact authenticated deployment generation."""
        self.start()
        if self._generation == prepared.generation:
            return
        try:
            frame = encode_load_artifact(
                generation=prepared.generation,
                model_bytes=prepared.model_bytes,
                model_kind=prepared.model_kind,
                runtime_abi=prepared.runtime_abi,
                max_uncompressed_bytes=prepared.max_uncompressed_bytes,
            )
        except IPCProtocolError as e:
            raise BackendProcessError(f"invalid isolated model bootstrap: {e}") from e
        deadline = time.monotonic() + self.control_timeout_s
        self._send_frame(frame, deadline=deadline, deadline_error=False)
        reply = self._receive_frame(deadline=deadline, deadline_error=False)
        try:
            header, _ = decode_frame(reply, max_payload_bytes=0)
            if header["op"] == "loaded":
                generation = decode_generation(reply, "loaded")
                if generation != prepared.generation:
                    raise IPCProtocolError("loaded generation mismatch")
                self._generation = generation
                return
            if header["op"] == "load_error":
                generation, code, message = decode_error(reply, "load_error", request=False)
                if generation != prepared.generation:
                    raise IPCProtocolError("load-error generation mismatch")
                raise BackendProcessError(
                    f"isolated inference worker rejected model ({code}): {message}"
                )
            raise IPCProtocolError("unexpected model-load response")
        except BackendProcessError:
            raise
        except IPCProtocolError as e:
            self.abort()
            self._increment("process_crashes")
            raise BackendProcessError("isolated inference worker rejected runtime snapshot") from e

    def execute(self, prepared, *, timeout_s: float, limits: dict[str, int]):
        self.load(prepared)
        self._request_id += 1
        request_id = self._request_id
        deadline = time.monotonic() + timeout_s
        try:
            frame = encode_predict(
                request_id=request_id,
                generation=prepared.generation,
                inputs=prepared.inputs,
            )
        except IPCProtocolError as e:
            raise ValidationError(f"invalid isolated inference input: {e}") from e
        self._send_frame(frame, deadline=deadline, deadline_error=True)
        reply = self._receive_frame(deadline=deadline, deadline_error=True)
        try:
            header, _ = decode_frame(
                reply,
                max_payload_bytes=limits["max_output_values"] * 4,
            )
            if header["op"] == "result":
                return decode_result(
                    reply,
                    expected_request_id=request_id,
                    expected_batch=len(prepared.inputs),
                    max_output_classes=limits["max_output_classes"],
                    max_output_values=limits["max_output_values"],
                )
            if header["op"] == "error":
                response_id, error_type, message = decode_error(reply, "error", request=True)
                if response_id != request_id:
                    raise IPCProtocolError("error response request ID mismatch")
                if error_type == "ValidationError":
                    raise ValidationError(message)
                raise BackendProcessError(
                    f"isolated backend error ({error_type}): {message}"
                )
            raise IPCProtocolError("unexpected inference response")
        except (ValidationError, BackendProcessError):
            raise
        except IPCProtocolError as e:
            self.abort()
            self._increment("process_crashes")
            raise BackendProcessError("isolated inference worker returned an invalid response") from e

    def is_alive(self) -> bool:
        process = self._process
        return bool(process is not None and process.is_alive() and self._channel is not None)

    def loaded_generation(self) -> int | None:
        return self._generation

    def close(self) -> None:
        channel = self._channel
        process = self._process
        if channel is not None and process is not None and process.is_alive():
            try:
                _socket_send_frame(
                    channel,
                    encode_simple("close"),
                    deadline=time.monotonic() + min(self.control_timeout_s, 0.1),
                )
                process.join(0.1)
            except (BrokenPipeError, EOFError, OSError, IPCProtocolError, TimeoutError):
                pass
        self.abort()


class BoundedInferenceExecutor:
    """Fixed-worker executor with bounded admission and optional process isolation.

    Queue deadlines govern how long a request may wait *before execution starts*.
    ``execution_mode='thread'`` preserves the lightweight reference behavior and never
    attempts unsafe thread cancellation. ``execution_mode='process'`` runs each worker's
    backend in a dedicated spawned subprocess; an execution timeout terminates that child
    and the next request starts a clean worker process.

    The process path transfers bounded bytes from the authenticated active-slot inode plus
    signed model-kind/ABI metadata over a strict raw-byte protocol; the child constructs
    its own runtime. Runtime objects are never pickled across the isolation boundary.
    """

    def __init__(
        self,
        engine,
        *,
        capacity: int = 64,
        policy: OverflowPolicy = OverflowPolicy.REJECT,
        workers: int = 1,
        execution_mode: ExecutionMode | str = ExecutionMode.THREAD,
        execution_timeout_s: float | None = None,
        process_control_timeout_s: float = 5.0,
        process_memory_limit_mb: int | None = None,
        process_nofile_limit: int = 64,
        process_restart_limit: int = 8,
        process_restart_window_s: float = 60.0,
    ):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1 or workers > 64:
            raise ValueError("workers must be an integer in [1,64]")
        self.engine = engine
        self.queue = BoundedInferenceQueue(capacity, policy)
        self.workers = workers
        self.execution_mode = ExecutionMode(execution_mode)
        if execution_timeout_s is not None and (
            isinstance(execution_timeout_s, bool)
            or not isinstance(execution_timeout_s, (int, float))
            or not math.isfinite(execution_timeout_s)
            or execution_timeout_s <= 0
        ):
            raise ValueError("execution_timeout_s must be finite and > 0")
        if (
            isinstance(process_control_timeout_s, bool)
            or not isinstance(process_control_timeout_s, (int, float))
            or not math.isfinite(process_control_timeout_s)
            or process_control_timeout_s <= 0
        ):
            raise ValueError("process_control_timeout_s must be finite and > 0")
        if self.execution_mode is ExecutionMode.PROCESS and execution_timeout_s is None:
            raise ValueError("process execution mode requires execution_timeout_s")
        if self.execution_mode is ExecutionMode.THREAD and execution_timeout_s is not None:
            raise ValueError("execution_timeout_s requires process execution mode")
        if process_memory_limit_mb is not None and (
            isinstance(process_memory_limit_mb, bool)
            or not isinstance(process_memory_limit_mb, int)
            or process_memory_limit_mb < 64
            or process_memory_limit_mb > 1_048_576
        ):
            raise ValueError("process_memory_limit_mb must be an integer in [64,1048576]")
        if self.execution_mode is ExecutionMode.THREAD and process_memory_limit_mb is not None:
            raise ValueError("process_memory_limit_mb requires process execution mode")
        if (
            isinstance(process_nofile_limit, bool)
            or not isinstance(process_nofile_limit, int)
            or process_nofile_limit < 16
            or process_nofile_limit > 1_048_576
        ):
            raise ValueError("process_nofile_limit must be an integer in [16,1048576]")
        if (
            isinstance(process_restart_limit, bool)
            or not isinstance(process_restart_limit, int)
            or process_restart_limit < 1
            or process_restart_limit > 10_000
        ):
            raise ValueError("process_restart_limit must be an integer in [1,10000]")
        if (
            isinstance(process_restart_window_s, bool)
            or not isinstance(process_restart_window_s, (int, float))
            or not math.isfinite(process_restart_window_s)
            or process_restart_window_s <= 0
            or process_restart_window_s > 86_400
        ):
            raise ValueError("process_restart_window_s must be finite and in (0,86400]")
        self.execution_timeout_s = None if execution_timeout_s is None else float(execution_timeout_s)
        self.process_control_timeout_s = float(process_control_timeout_s)
        self.process_memory_limit_mb = process_memory_limit_mb
        self.process_memory_limit_bytes = (
            None if process_memory_limit_mb is None else process_memory_limit_mb * 1024 * 1024
        )
        self.process_nofile_limit = process_nofile_limit
        self.process_restart_limit = process_restart_limit
        self.process_restart_window_s = float(process_restart_window_s)
        self._isolation_limits: dict[str, int] | None = None
        if self.execution_mode is ExecutionMode.PROCESS:
            if not all(
                callable(getattr(self.engine, name, None))
                for name in (
                    "prepare_isolated",
                    "isolated_runtime_snapshot",
                    "complete",
                    "record_failure",
                    "observe_latency_ms",
                    "isolation_limits",
                )
            ):
                raise ValueError("engine does not support authenticated process isolation")
            limits = self.engine.isolation_limits()
            required = {
                "max_batch",
                "max_input_features",
                "max_input_values",
                "max_output_classes",
                "max_output_values",
                "max_model_transfer_bytes",
            }
            if not isinstance(limits, dict) or set(limits) != required or any(
                isinstance(limits[name], bool)
                or not isinstance(limits[name], int)
                or limits[name] < 1
                for name in required
            ):
                raise ValueError("engine returned invalid process-isolation limits")
            self._isolation_limits = dict(limits)
        self._state_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._threads: list[threading.Thread] = []
        self._completed = 0
        self._failed = 0
        self._process_starts = 0
        self._process_restarts = 0
        self._process_crashes = 0
        self._process_restart_throttled = 0
        self._execution_timeouts = 0
        self._counter_lock = threading.Lock()
        self._process_runners: list[_ProcessInferenceRunner] = []
        self._restart_budget = _RollingRestartBudget(
            limit=self.process_restart_limit,
            window_s=self.process_restart_window_s,
            on_throttled=lambda: self._increment_process_counter("process_restart_throttled"),
        )

    def _increment_process_counter(self, name: str) -> None:
        with self._counter_lock:
            if name == "process_starts":
                self._process_starts += 1
            elif name == "process_restarts":
                self._process_restarts += 1
            elif name == "process_crashes":
                self._process_crashes += 1
            elif name == "process_restart_throttled":
                self._process_restart_throttled += 1
            elif name == "execution_timeouts":
                self._execution_timeouts += 1

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise ExecutorClosed("inference executor is closed")
            if self._started:
                return
            self._started = True
            if self.execution_mode is ExecutionMode.PROCESS:
                self._process_runners = [
                    _ProcessInferenceRunner(
                        control_timeout_s=self.process_control_timeout_s,
                        memory_limit_bytes=self.process_memory_limit_bytes,
                        nofile_limit=self.process_nofile_limit,
                        restart_limit=self.process_restart_limit,
                        restart_window_s=self.process_restart_window_s,
                        restart_budget=self._restart_budget,
                        ipc_limits=self._isolation_limits,
                        counters=self._increment_process_counter,
                    )
                    for _ in range(self.workers)
                ]
                try:
                    for runner in self._process_runners:
                        runner.start()
                except Exception:
                    for runner in self._process_runners:
                        runner.close()
                    self._process_runners = []
                    self._started = False
                    raise
            for index in range(self.workers):
                thread = threading.Thread(
                    target=self._worker,
                    args=(index,),
                    name=f"edgeai-inference-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

    def _enqueue(self, task: _InferenceTask, deadline_s: float | None) -> None:
        self.start()
        with self._state_lock:
            if self._closed:
                raise ExecutorClosed("inference executor is closed")
            accepted, evicted = self.queue.push_with_evicted(task, deadline_s=deadline_s)
            if not accepted:
                raise QueueFull("inference queue dropped newest request")
            if evicted is not None:
                evicted.payload.complete(error=QueueDropped("inference queue dropped oldest request"))

    def submit(self, payload, *, deadline_s: float | None = None):
        task = _InferenceTask(payload)
        self._enqueue(task, deadline_s)
        if deadline_s is None:
            task.done.wait()
        elif not task.done.wait(deadline_s):
            if task.expire_if_not_started():
                self.queue.expire_payload(task)
            else:
                # The worker met the queue-start deadline. In process mode, the hard
                # execution deadline is enforced by the worker supervisor. In thread
                # mode we retain the safe no-cancellation behavior.
                task.done.wait()
        if task.error is not None:
            raise task.error
        return task.result

    async def submit_async(self, payload, *, deadline_s: float | None = None):
        """Coroutine-friendly admission that does not consume a web-server worker thread."""
        loop = asyncio.get_running_loop()
        async_done = asyncio.Event()
        task = _InferenceTask(
            payload,
            notify=lambda: loop.call_soon_threadsafe(async_done.set),
        )
        self._enqueue(task, deadline_s)
        try:
            if deadline_s is None:
                await async_done.wait()
            else:
                try:
                    await asyncio.wait_for(async_done.wait(), timeout=deadline_s)
                except TimeoutError:
                    if task.expire_if_not_started():
                        self.queue.expire_payload(task)
                    else:
                        # Execution started before the queue deadline. A process worker
                        # applies its own execution timeout; a thread worker cannot be
                        # cancelled safely once native inference begins.
                        await async_done.wait()
        except asyncio.CancelledError:
            if task.cancel_if_not_started():
                self.queue.expire_payload(task)
            raise
        if task.error is not None:
            raise task.error
        return task.result

    def _process_predict(self, index: int, payload):
        prepared = self.engine.prepare_isolated(payload)
        started = time.perf_counter()
        try:
            logits = self._process_runners[index].execute(
                prepared,
                timeout_s=self.execution_timeout_s,
                limits=self._isolation_limits,
            )
            return self.engine.complete(prepared, logits)
        except Exception:
            self.engine.record_failure()
            raise
        finally:
            self.engine.observe_latency_ms(started)

    def _worker(self, index: int) -> None:
        while True:
            item, expired = self.queue.pop_with_expired(wait_s=0.1)
            for expired_item in expired:
                expired_item.payload.complete(
                    error=DeadlineExceeded("inference request expired before execution")
                )
            if item is None:
                with self._state_lock:
                    if self._closed:
                        return
                continue
            task = item.payload
            if not task.begin():
                continue
            try:
                if self.execution_mode is ExecutionMode.PROCESS:
                    result = self._process_predict(index, task.payload)
                else:
                    result = self.engine.predict(task.payload)
            except BaseException as e:
                task.complete(error=e)
                with self._counter_lock:
                    self._failed += 1
            else:
                task.complete(result=result)
                with self._counter_lock:
                    self._completed += 1

    def ensure_ready(self) -> dict:
        """Ensure executor workers are available and return an operational snapshot.

        In process mode this self-heals an idle child that was killed externally. It does
        not interrupt a currently running request and does not load a model eagerly.
        """
        self.start()
        with self._state_lock:
            if self._closed:
                raise ExecutorClosed("inference executor is closed")
            runners = list(self._process_runners)
        if self.execution_mode is ExecutionMode.PROCESS:
            for runner in runners:
                runner.start()
            try:
                snapshot = self.engine.isolated_runtime_snapshot()
            except DeploymentError:
                # No active deployment is a model-readiness condition, not an executor
                # transport failure. The service's engine readiness check reports it.
                snapshot = None
            if snapshot is not None:
                for runner in runners:
                    runner.load(snapshot)
        return self.stats()

    def close(self, *, wait: bool = True, timeout_s: float | None = None) -> bool:
        """Close admission and optionally wait a bounded time for running inference.

        Thread mode never attempts unsafe asynchronous cancellation. Process mode first
        terminates child backends, allowing a worker blocked in IPC to unwind before the
        executor threads are joined.
        """
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError("timeout_s must be finite and >= 0")
        with self._state_lock:
            if self._closed:
                threads = list(self._threads)
                runners = list(self._process_runners)
            else:
                self._closed = True
                queued = self.queue.drain()
                for item in queued:
                    item.payload.complete(error=ExecutorClosed("inference executor is closed"))
                threads = list(self._threads)
                runners = list(self._process_runners)
        self.queue.wake_all()
        if self.execution_mode is ExecutionMode.PROCESS:
            for runner in runners:
                runner.close()
        if wait:
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            for thread in threads:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                thread.join(remaining)
        return all(not thread.is_alive() for thread in threads)

    def stats(self):
        stats = self.queue.stats()
        with self._counter_lock:
            stats.update(
                {
                    "completed": self._completed,
                    "failed": self._failed,
                    "process_starts": self._process_starts,
                    "process_restarts": self._process_restarts,
                    "process_crashes": self._process_crashes,
                    "process_restart_throttled": self._process_restart_throttled,
                    "execution_timeouts": self._execution_timeouts,
                }
            )
        with self._state_lock:
            runners = list(self._process_runners)
            stats.update(
                {
                    "workers": self.workers,
                    "closed": self._closed,
                    "execution_mode": self.execution_mode.value,
                    "process_memory_limit_mb": self.process_memory_limit_mb,
                    "process_nofile_limit": self.process_nofile_limit,
                    "process_restart_limit": self.process_restart_limit,
                    "process_restart_window_s": self.process_restart_window_s,
                    "process_restart_budget_scope": "executor",
                    "process_restarts_in_window": self._restart_budget.current(),
                    "process_workers_alive": sum(runner.is_alive() for runner in runners),
                    "process_workers_loaded": sum(
                        runner.loaded_generation() is not None for runner in runners
                    ),
                    "process_ipc_protocol": "raw-v1"
                    if self.execution_mode is ExecutionMode.PROCESS
                    else "none",
                }
            )
        return stats
