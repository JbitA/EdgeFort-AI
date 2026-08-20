from __future__ import annotations

"""Killable deployment-time model qualification.

Candidate model parsing and benchmarking are intentionally performed outside the
lifecycle/control process.  The parent authenticates model bytes before this module is
called; this layer contains parser/backend hangs, crashes and address-space blowups.
Runtime messages use the same strict non-pickle IPC protocol as isolated inference.
"""

from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import socket
import time

from .errors import HealthGateError
from .ipc import (
    IPCProtocolError,
    MAX_HEADER_BYTES,
    decode_error,
    decode_generation,
    decode_load_artifact,
    decode_metrics,
    decode_qualification,
    decode_ready,
    encode_error,
    encode_generation,
    encode_load_artifact,
    encode_metrics,
    encode_qualification,
    encode_ready,
)
from .runtime import (
    _socket_recv_frame,
    _socket_send_frame,
)
from .sandbox import apply_process_sandbox


@dataclass(frozen=True)
class QualificationLimits:
    timeout_s: float = 30.0
    memory_limit_mb: int | None = None
    max_validation_rows: int = 100_000
    max_validation_features: int = 1_000_000
    max_validation_values: int = 10_000_000
    iterations: int = 40
    batch: int = 32
    nofile_limit: int = 64

    def validate(self) -> "QualificationLimits":
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("qualification timeout must be finite and > 0")
        if self.memory_limit_mb is not None and (
            isinstance(self.memory_limit_mb, bool)
            or not isinstance(self.memory_limit_mb, int)
            or not 64 <= self.memory_limit_mb <= 1_048_576
        ):
            raise ValueError("qualification memory limit must be an integer in [64,1048576]")
        for name, value, maximum in (
            ("max_validation_rows", self.max_validation_rows, 10_000_000),
            ("max_validation_features", self.max_validation_features, 10_000_000),
            ("max_validation_values", self.max_validation_values, 1_000_000_000),
            ("iterations", self.iterations, 100_000),
            ("batch", self.batch, 10_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"invalid qualification {name}")
        if (
            isinstance(self.nofile_limit, bool)
            or not isinstance(self.nofile_limit, int)
            or not 16 <= self.nofile_limit <= 1_048_576
        ):
            raise ValueError("invalid qualification nofile_limit")
        return self

    @property
    def memory_limit_bytes(self) -> int | None:
        return None if self.memory_limit_mb is None else self.memory_limit_mb * 1024 * 1024


def _qualification_worker(
    channel: socket.socket,
    *,
    max_model_bytes: int,
    limits: QualificationLimits,
) -> None:
    runtime = None
    max_validation_payload = limits.max_validation_values * 4 + limits.max_validation_rows * 8
    max_incoming = MAX_HEADER_BYTES + 1 + max(max_model_bytes, max_validation_payload)
    try:
        try:
            apply_process_sandbox(
                memory_limit_bytes=limits.memory_limit_bytes,
                nofile_limit=limits.nofile_limit,
                clear_environment=True,
                no_new_privs=True,
            )
        except Exception as e:
            _socket_send_frame(
                channel,
                encode_error(
                    "qualification_error",
                    request_id=None,
                    generation=0,
                    code=type(e).__name__,
                    message=str(e)[:4096] or "resource setup failed",
                ),
                deadline=time.monotonic() + 1.0,
            )
            return
        _socket_send_frame(
            channel,
            encode_ready(os.getpid()),
            deadline=time.monotonic() + 1.0,
        )
        model_frame = _socket_recv_frame(channel, max_frame_bytes=max_incoming, deadline=None)
        try:
            generation, model_bytes, model_kind, runtime_abi, max_uncompressed = decode_load_artifact(
                model_frame, max_model_bytes=max_model_bytes
            )
            from .health import load_runtime
            import io

            runtime = load_runtime(
                io.BytesIO(model_bytes),
                model_kind,
                runtime_abi,
                max_uncompressed_bytes=max_uncompressed,
            )
            _socket_send_frame(
                channel,
                encode_generation("loaded", generation),
                deadline=time.monotonic() + 1.0,
            )
        except Exception as e:
            _socket_send_frame(
                channel,
                encode_error(
                    "qualification_error",
                    request_id=None,
                    generation=0,
                    code=type(e).__name__,
                    message=str(e)[:4096] or "candidate load failed",
                ),
                deadline=time.monotonic() + 1.0,
            )
            return

        qualification_frame = _socket_recv_frame(
            channel, max_frame_bytes=max_incoming, deadline=None
        )
        try:
            x, y, iterations, batch = decode_qualification(
                qualification_frame,
                max_validation_rows=limits.max_validation_rows,
                max_validation_features=limits.max_validation_features,
                max_validation_values=limits.max_validation_values,
                max_iterations=limits.iterations,
            )
            from .health import benchmark

            metrics = benchmark(runtime, x, y, iterations=iterations, batch=batch)
            _socket_send_frame(
                channel,
                encode_metrics(metrics),
                deadline=time.monotonic() + 1.0,
            )
        except Exception as e:
            _socket_send_frame(
                channel,
                encode_error(
                    "qualification_error",
                    request_id=None,
                    generation=0,
                    code=type(e).__name__,
                    message=str(e)[:4096] or "candidate qualification failed",
                ),
                deadline=time.monotonic() + 1.0,
            )
    except Exception:
        # A malformed parent message or broken channel is fatal to this disposable worker.
        return
    finally:
        try:
            channel.close()
        except Exception:
            pass


def _terminate(process: mp.Process) -> None:
    if not process.is_alive():
        process.join(timeout=0.1)
        return
    process.terminate()
    process.join(timeout=0.5)
    if process.is_alive():
        kill = getattr(process, "kill", None)
        if kill is not None:
            kill()
            process.join(timeout=0.5)


def qualify_model_bytes(
    *,
    model_bytes: bytes,
    model_kind: str,
    runtime_abi: str,
    max_uncompressed_bytes: int,
    x_validation,
    y_validation,
    limits: QualificationLimits | None = None,
):
    """Load and benchmark one authenticated model in a killable spawned process."""
    resolved = (limits or QualificationLimits()).validate()
    if mp.current_process().daemon:
        raise HealthGateError("isolated qualification cannot start from a daemon process")
    if not isinstance(model_bytes, bytes) or not model_bytes:
        raise HealthGateError("candidate model bytes must be non-empty")
    if (
        isinstance(max_uncompressed_bytes, bool)
        or not isinstance(max_uncompressed_bytes, int)
        or max_uncompressed_bytes < 1024
    ):
        raise HealthGateError("invalid candidate model size bound")
    if len(model_bytes) > max_uncompressed_bytes:
        raise HealthGateError("candidate artifact exceeds qualification transfer bound")

    try:
        validation_frame = encode_qualification(
            inputs=x_validation,
            labels=y_validation,
            iterations=resolved.iterations,
            batch=resolved.batch,
            max_validation_rows=resolved.max_validation_rows,
            max_validation_features=resolved.max_validation_features,
            max_validation_values=resolved.max_validation_values,
        )
        model_frame = encode_load_artifact(
            generation=0,
            model_bytes=model_bytes,
            model_kind=model_kind,
            runtime_abi=runtime_abi,
            max_uncompressed_bytes=max_uncompressed_bytes,
        )
    except (IPCProtocolError, ValueError) as e:
        raise HealthGateError(f"invalid qualification input: {e}") from e

    parent, child = socket.socketpair()
    context = mp.get_context("spawn")
    process = context.Process(
        target=_qualification_worker,
        kwargs={
            "channel": child,
            "max_model_bytes": max_uncompressed_bytes,
            "limits": resolved,
        },
        name="edgeai-health-qualification",
        daemon=True,
    )
    deadline = time.monotonic() + float(resolved.timeout_s)
    try:
        process.start()
        child.close()
        max_control = MAX_HEADER_BYTES + 1 + 4096
        ready = _socket_recv_frame(parent, max_frame_bytes=max_control, deadline=deadline)
        try:
            decode_ready(ready)
        except IPCProtocolError:
            try:
                _, code, message = decode_error(ready, "qualification_error", request=False)
            except IPCProtocolError as e:
                raise HealthGateError("qualification worker sent invalid startup response") from e
            raise HealthGateError(f"qualification worker startup failed: {code}: {message}")

        _socket_send_frame(parent, model_frame, deadline=deadline)
        load_response = _socket_recv_frame(parent, max_frame_bytes=max_control, deadline=deadline)
        try:
            decode_generation(load_response, "loaded")
        except IPCProtocolError:
            try:
                _, code, message = decode_error(
                    load_response, "qualification_error", request=False
                )
            except IPCProtocolError as e:
                raise HealthGateError("qualification worker sent invalid load response") from e
            raise HealthGateError(f"candidate load failed: {code}: {message}")

        _socket_send_frame(parent, validation_frame, deadline=deadline)
        response = _socket_recv_frame(parent, max_frame_bytes=max_control, deadline=deadline)
        try:
            metrics = decode_metrics(response)
        except IPCProtocolError:
            try:
                _, code, message = decode_error(response, "qualification_error", request=False)
            except IPCProtocolError as e:
                raise HealthGateError("qualification worker sent invalid metrics response") from e
            raise HealthGateError(f"candidate qualification failed: {code}: {message}")
        return metrics
    except (TimeoutError, socket.timeout) as e:
        raise HealthGateError("candidate qualification exceeded hard wall-clock deadline") from e
    except (EOFError, BrokenPipeError, ConnectionError, OSError) as e:
        raise HealthGateError("candidate qualification worker crashed or disconnected") from e
    finally:
        try:
            parent.close()
        except Exception:
            pass
        try:
            child.close()
        except Exception:
            pass
        _terminate(process)
