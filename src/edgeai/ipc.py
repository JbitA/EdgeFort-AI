from __future__ import annotations

"""Strict binary-safe IPC framing for process-isolated inference.

The protocol deliberately never uses pickle for runtime messages.  Frames are a small
canonical JSON header followed by an optional raw payload.  The surrounding socket
transport prefixes each frame with an unsigned 64-bit network-order length so receivers
can reject oversized messages *before* allocating the frame body.
"""

import json
import struct
from typing import Mapping

import numpy as np

from .errors import ValidationError

IPC_MAGIC = "EDGEAI-IPC"
IPC_VERSION = 1
MAX_HEADER_BYTES = 4096
MAX_ERROR_TEXT_BYTES = 4096
_FRAME_LENGTH = struct.Struct("!Q")


class IPCProtocolError(ValueError):
    """Peer sent an invalid or unsupported inference IPC frame."""


def _reject_constant(value: str):
    raise IPCProtocolError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IPCProtocolError(f"duplicate IPC header field {key!r}")
        result[key] = value
    return result


def _uint(name: str, value, maximum: int = (1 << 63) - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise IPCProtocolError(f"invalid IPC integer field {name}")
    return value


def _text(name: str, value, *, max_bytes: int = 1024) -> str:
    if not isinstance(value, str) or not value:
        raise IPCProtocolError(f"invalid IPC text field {name}")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as e:
        raise IPCProtocolError(f"invalid UTF-8 IPC field {name}") from e
    if len(encoded) > max_bytes:
        raise IPCProtocolError(f"IPC text field {name} is too large")
    return value


def encode_frame(op: str, fields: Mapping[str, object] | None = None, payload: bytes = b"") -> bytes:
    """Encode one deterministic protocol frame without Python object serialization."""
    if not isinstance(op, str) or not op:
        raise IPCProtocolError("invalid IPC operation")
    try:
        encoded_op = op.encode("ascii", "strict")
    except UnicodeEncodeError as e:
        raise IPCProtocolError("invalid IPC operation") from e
    if len(encoded_op) > 32:
        raise IPCProtocolError("invalid IPC operation")
    if not isinstance(payload, bytes):
        raise IPCProtocolError("IPC payload must be bytes")
    header = {
        "magic": IPC_MAGIC,
        "op": op,
        "payload_bytes": len(payload),
        "v": IPC_VERSION,
    }
    if fields:
        for key, value in fields.items():
            if key in header:
                raise IPCProtocolError(f"reserved IPC header field {key!r}")
            if not isinstance(key, str) or not key:
                raise IPCProtocolError("IPC header keys must be non-empty strings")
            header[key] = value
    try:
        encoded = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as e:
        raise IPCProtocolError(f"IPC header is not canonical JSON: {e}") from e
    if len(encoded) > MAX_HEADER_BYTES:
        raise IPCProtocolError("IPC header exceeds configured bound")
    return encoded + b"\n" + payload


def decode_frame(frame: bytes, *, max_payload_bytes: int) -> tuple[dict[str, object], bytes]:
    if not isinstance(frame, bytes):
        raise IPCProtocolError("IPC frame must be bytes")
    if isinstance(max_payload_bytes, bool) or not isinstance(max_payload_bytes, int) or max_payload_bytes < 0:
        raise ValueError("max_payload_bytes must be a non-negative integer")
    newline = frame.find(b"\n", 0, MAX_HEADER_BYTES + 1)
    if newline < 0:
        raise IPCProtocolError("IPC frame missing bounded header terminator")
    header_bytes = frame[:newline]
    if not header_bytes:
        raise IPCProtocolError("IPC frame has empty header")
    try:
        header = json.loads(
            header_bytes.decode("ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise IPCProtocolError(f"invalid IPC JSON header: {e}") from e
    if not isinstance(header, dict):
        raise IPCProtocolError("IPC header must be an object")
    for key in ("magic", "op", "payload_bytes", "v"):
        if key not in header:
            raise IPCProtocolError(f"missing IPC header field {key}")
    if header["magic"] != IPC_MAGIC or header["v"] != IPC_VERSION:
        raise IPCProtocolError("unsupported IPC protocol")
    _text("op", header["op"], max_bytes=32)
    payload_bytes = _uint("payload_bytes", header["payload_bytes"], maximum=max_payload_bytes)
    payload = frame[newline + 1 :]
    if len(payload) != payload_bytes:
        raise IPCProtocolError("IPC payload length mismatch")
    return header, payload


def require_fields(header: Mapping[str, object], *fields: str) -> None:
    expected = {"magic", "op", "payload_bytes", "v", *fields}
    actual = set(header)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise IPCProtocolError(f"IPC fields mismatch; missing={missing}, unknown={unknown}")


def encode_ready(pid: int) -> bytes:
    return encode_frame("ready", {"pid": _uint("pid", pid, maximum=(1 << 31) - 1)})


def decode_ready(frame: bytes) -> int:
    header, payload = decode_frame(frame, max_payload_bytes=0)
    require_fields(header, "pid")
    if header["op"] != "ready" or payload:
        raise IPCProtocolError("expected ready frame")
    pid = _uint("pid", header["pid"], maximum=(1 << 31) - 1)
    if pid == 0:
        raise IPCProtocolError("invalid worker pid")
    return pid


def encode_simple(op: str) -> bytes:
    return encode_frame(op)


def decode_simple(frame: bytes, op: str) -> None:
    header, payload = decode_frame(frame, max_payload_bytes=0)
    require_fields(header)
    if header["op"] != op or payload:
        raise IPCProtocolError(f"expected {op} frame")


def encode_generation(op: str, generation: int) -> bytes:
    return encode_frame(op, {"generation": _uint("generation", generation)})


def decode_generation(frame: bytes, op: str) -> int:
    header, payload = decode_frame(frame, max_payload_bytes=0)
    require_fields(header, "generation")
    if header["op"] != op or payload:
        raise IPCProtocolError(f"expected {op} frame")
    return _uint("generation", header["generation"])


def encode_error(op: str, *, request_id: int | None, generation: int | None, code: str, message: str) -> bytes:
    fields: dict[str, object] = {
        "code": _text("code", code, max_bytes=64),
        "message": _text("message", message or "backend failure", max_bytes=MAX_ERROR_TEXT_BYTES),
    }
    if request_id is not None:
        fields["request_id"] = _uint("request_id", request_id)
    if generation is not None:
        fields["generation"] = _uint("generation", generation)
    return encode_frame(op, fields)


def decode_error(frame: bytes, op: str, *, request: bool) -> tuple[int, str, str] | tuple[int, str, str]:
    header, payload = decode_frame(frame, max_payload_bytes=0)
    id_field = "request_id" if request else "generation"
    require_fields(header, id_field, "code", "message")
    if header["op"] != op or payload:
        raise IPCProtocolError(f"expected {op} frame")
    identifier = _uint(id_field, header[id_field])
    code = _text("code", header["code"], max_bytes=64)
    message = _text("message", header["message"], max_bytes=MAX_ERROR_TEXT_BYTES)
    return identifier, code, message


def encode_startup_error(code: str, message: str) -> bytes:
    return encode_frame(
        "startup_error",
        {
            "code": _text("code", code, max_bytes=64),
            "message": _text("message", message or "worker startup failed", max_bytes=MAX_ERROR_TEXT_BYTES),
        },
    )


def decode_startup_error(frame: bytes) -> tuple[str, str]:
    header, payload = decode_frame(frame, max_payload_bytes=0)
    require_fields(header, "code", "message")
    if header["op"] != "startup_error" or payload:
        raise IPCProtocolError("expected startup_error frame")
    return (
        _text("code", header["code"], max_bytes=64),
        _text("message", header["message"], max_bytes=MAX_ERROR_TEXT_BYTES),
    )


def encode_load_artifact(
    *,
    generation: int,
    model_bytes: bytes,
    model_kind: str,
    runtime_abi: str,
    max_uncompressed_bytes: int,
) -> bytes:
    if not isinstance(model_bytes, bytes) or not model_bytes:
        raise IPCProtocolError("model artifact payload must be non-empty bytes")
    return encode_frame(
        "load_artifact",
        {
            "generation": _uint("generation", generation),
            "max_uncompressed_bytes": _uint("max_uncompressed_bytes", max_uncompressed_bytes),
            "model_kind": _text("model_kind", model_kind, max_bytes=256),
            "runtime_abi": _text("runtime_abi", runtime_abi, max_bytes=256),
        },
        model_bytes,
    )


def decode_load_artifact(frame: bytes, *, max_model_bytes: int):
    header, payload = decode_frame(frame, max_payload_bytes=max_model_bytes)
    require_fields(header, "generation", "max_uncompressed_bytes", "model_kind", "runtime_abi")
    if header["op"] != "load_artifact" or not payload:
        raise IPCProtocolError("expected load_artifact frame")
    generation = _uint("generation", header["generation"])
    max_uncompressed = _uint("max_uncompressed_bytes", header["max_uncompressed_bytes"])
    if max_uncompressed < 1024 or max_uncompressed > max_model_bytes:
        raise IPCProtocolError("invalid model uncompressed bound")
    return (
        generation,
        payload,
        _text("model_kind", header["model_kind"], max_bytes=256),
        _text("runtime_abi", header["runtime_abi"], max_bytes=256),
        max_uncompressed,
    )


def _float32_payload(array) -> tuple[np.ndarray, bytes]:
    try:
        normalized = np.asarray(array, dtype=np.dtype("<f4"), order="C")
    except (TypeError, ValueError, OverflowError) as e:
        raise IPCProtocolError("IPC tensor cannot be represented as float32") from e
    if normalized.ndim != 2:
        raise IPCProtocolError("IPC tensor must be rank-2")
    return normalized, normalized.tobytes(order="C")


def encode_predict(*, request_id: int, generation: int, inputs) -> bytes:
    array, payload = _float32_payload(inputs)
    return encode_frame(
        "predict",
        {
            "cols": _uint("cols", int(array.shape[1]), maximum=(1 << 31) - 1),
            "generation": _uint("generation", generation),
            "request_id": _uint("request_id", request_id),
            "rows": _uint("rows", int(array.shape[0]), maximum=(1 << 31) - 1),
        },
        payload,
    )


def decode_predict(
    frame: bytes,
    *,
    max_batch: int,
    max_input_features: int,
    max_input_values: int,
):
    max_payload = max_input_values * 4
    header, payload = decode_frame(frame, max_payload_bytes=max_payload)
    require_fields(header, "cols", "generation", "request_id", "rows")
    if header["op"] != "predict":
        raise IPCProtocolError("expected predict frame")
    request_id = _uint("request_id", header["request_id"])
    generation = _uint("generation", header["generation"])
    rows = _uint("rows", header["rows"], maximum=max_batch)
    cols = _uint("cols", header["cols"], maximum=max_input_features)
    if rows < 1 or cols < 1 or rows * cols > max_input_values:
        raise IPCProtocolError("IPC input tensor outside configured bounds")
    expected = rows * cols * 4
    if len(payload) != expected:
        raise IPCProtocolError("IPC input tensor byte length mismatch")
    array = np.frombuffer(payload, dtype=np.dtype("<f4")).reshape((rows, cols))
    if not np.isfinite(array).all():
        raise IPCProtocolError("IPC input tensor contains non-finite values")
    # Copy so the ndarray does not retain the potentially much larger frame bytes object.
    return request_id, generation, array.copy()


def encode_result(*, request_id: int, logits) -> bytes:
    array, payload = _float32_payload(logits)
    return encode_frame(
        "result",
        {
            "cols": _uint("cols", int(array.shape[1]), maximum=(1 << 31) - 1),
            "request_id": _uint("request_id", request_id),
            "rows": _uint("rows", int(array.shape[0]), maximum=(1 << 31) - 1),
        },
        payload,
    )


def decode_result(
    frame: bytes,
    *,
    expected_request_id: int,
    expected_batch: int,
    max_output_classes: int,
    max_output_values: int,
) -> np.ndarray:
    header, payload = decode_frame(frame, max_payload_bytes=max_output_values * 4)
    require_fields(header, "cols", "request_id", "rows")
    if header["op"] != "result":
        raise IPCProtocolError("expected result frame")
    request_id = _uint("request_id", header["request_id"])
    if request_id != expected_request_id:
        raise IPCProtocolError("IPC result request ID mismatch")
    rows = _uint("rows", header["rows"], maximum=expected_batch)
    cols = _uint("cols", header["cols"], maximum=max_output_classes)
    if rows != expected_batch or cols < 1 or rows * cols > max_output_values:
        raise IPCProtocolError("IPC output tensor outside configured bounds")
    expected = rows * cols * 4
    if len(payload) != expected:
        raise IPCProtocolError("IPC output tensor byte length mismatch")
    array = np.frombuffer(payload, dtype=np.dtype("<f4")).reshape((rows, cols))
    if not np.isfinite(array).all():
        raise IPCProtocolError("IPC output tensor contains non-finite values")
    return array.copy()



def encode_qualification(
    *,
    inputs,
    labels,
    iterations: int = 40,
    batch: int = 32,
    max_validation_rows: int = (1 << 31) - 1,
    max_validation_features: int = (1 << 31) - 1,
    max_validation_values: int = (1 << 63) - 1,
) -> bytes:
    """Encode a validation dataset without pickle, enforcing bounds before serialization."""
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (max_validation_rows, max_validation_features, max_validation_values)
    ):
        raise ValueError("qualification encode limits must be positive integers")
    try:
        x = np.asarray(inputs, dtype=np.dtype("<f4"), order="C")
    except (TypeError, ValueError, OverflowError) as e:
        raise IPCProtocolError("qualification inputs cannot be represented as float32") from e
    y_raw = np.asarray(labels)
    if x.ndim != 2 or y_raw.ndim != 1 or len(x) != len(y_raw) or len(x) < 1:
        raise IPCProtocolError("qualification dataset shape mismatch")
    rows, cols = int(x.shape[0]), int(x.shape[1])
    if (
        rows > max_validation_rows
        or cols < 1
        or cols > max_validation_features
        or rows * cols > max_validation_values
    ):
        raise IPCProtocolError("qualification dataset outside configured bounds")
    if y_raw.dtype.kind not in {"i", "u"}:
        raise IPCProtocolError("qualification labels must be integers")
    if not np.isfinite(x).all():
        raise IPCProtocolError("qualification inputs contain non-finite values")
    try:
        y = np.asarray(y_raw, dtype=np.dtype("<i8"), order="C")
    except (TypeError, ValueError, OverflowError) as e:
        raise IPCProtocolError("qualification labels cannot be represented as int64") from e
    if np.any(y < 0):
        raise IPCProtocolError("qualification labels must be non-negative")
    x_payload = x.tobytes(order="C")
    y_payload = y.tobytes(order="C")
    return encode_frame(
        "qualify",
        {
            "batch": _uint("batch", batch, maximum=(1 << 31) - 1),
            "cols": _uint("cols", int(x.shape[1]), maximum=(1 << 31) - 1),
            "iterations": _uint("iterations", iterations, maximum=100_000),
            "rows": _uint("rows", int(x.shape[0]), maximum=(1 << 31) - 1),
            "x_bytes": len(x_payload),
            "y_bytes": len(y_payload),
        },
        x_payload + y_payload,
    )


def decode_qualification(
    frame: bytes,
    *,
    max_validation_rows: int,
    max_validation_features: int,
    max_validation_values: int,
    max_iterations: int = 10_000,
):
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (
            max_validation_rows,
            max_validation_features,
            max_validation_values,
            max_iterations,
        )
    ):
        raise ValueError("qualification decode limits must be positive integers")
    max_payload = max_validation_values * 4 + max_validation_rows * 8
    header, payload = decode_frame(frame, max_payload_bytes=max_payload)
    require_fields(header, "batch", "cols", "iterations", "rows", "x_bytes", "y_bytes")
    if header["op"] != "qualify":
        raise IPCProtocolError("expected qualify frame")
    rows = _uint("rows", header["rows"], maximum=max_validation_rows)
    cols = _uint("cols", header["cols"], maximum=max_validation_features)
    iterations = _uint("iterations", header["iterations"], maximum=max_iterations)
    batch = _uint("batch", header["batch"], maximum=max_validation_rows)
    if rows < 1 or cols < 1 or rows * cols > max_validation_values:
        raise IPCProtocolError("qualification dataset outside configured bounds")
    if iterations < 1 or batch < 1:
        raise IPCProtocolError("qualification benchmark parameters outside configured bounds")
    x_bytes = _uint("x_bytes", header["x_bytes"], maximum=max_validation_values * 4)
    y_bytes = _uint("y_bytes", header["y_bytes"], maximum=max_validation_rows * 8)
    expected_x = rows * cols * 4
    expected_y = rows * 8
    if x_bytes != expected_x or y_bytes != expected_y or len(payload) != x_bytes + y_bytes:
        raise IPCProtocolError("qualification payload byte length mismatch")
    x = np.frombuffer(payload[:x_bytes], dtype=np.dtype("<f4")).reshape((rows, cols))
    y = np.frombuffer(payload[x_bytes:], dtype=np.dtype("<i8"))
    if not np.isfinite(x).all() or np.any(y < 0):
        raise IPCProtocolError("qualification dataset contains invalid values")
    return x.copy(), y.copy(), iterations, batch


def _finite_metric(name: str, value, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IPCProtocolError(f"invalid IPC metric field {name}")
    result = float(value)
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise IPCProtocolError(f"invalid IPC metric field {name}")
    return result


def encode_metrics(metrics) -> bytes:
    metrics.validate()
    return encode_frame(
        "metrics",
        {
            "accuracy": float(metrics.accuracy),
            "ece": float(metrics.ece),
            "max_rss_mb": float(metrics.max_rss_mb),
            "p95_latency_ms": float(metrics.p95_latency_ms),
        },
    )


def decode_metrics(frame: bytes):
    from .types import ModelMetrics

    header, payload = decode_frame(frame, max_payload_bytes=0)
    require_fields(header, "accuracy", "ece", "max_rss_mb", "p95_latency_ms")
    if header["op"] != "metrics" or payload:
        raise IPCProtocolError("expected metrics frame")
    return ModelMetrics(
        accuracy=_finite_metric("accuracy", header["accuracy"], minimum=0.0, maximum=1.0),
        ece=_finite_metric("ece", header["ece"], minimum=0.0, maximum=1.0),
        p95_latency_ms=_finite_metric(
            "p95_latency_ms", header["p95_latency_ms"], minimum=0.0, maximum=1e12
        ),
        max_rss_mb=_finite_metric("max_rss_mb", header["max_rss_mb"], minimum=0.0, maximum=1e12),
    )

def frame_length_prefix(size: int) -> bytes:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0 or size > (1 << 63) - 1:
        raise IPCProtocolError("invalid IPC frame size")
    return _FRAME_LENGTH.pack(size)


def parse_frame_length(prefix: bytes) -> int:
    if len(prefix) != _FRAME_LENGTH.size:
        raise IPCProtocolError("invalid IPC frame-length prefix")
    return _FRAME_LENGTH.unpack(prefix)[0]


FRAME_PREFIX_BYTES = _FRAME_LENGTH.size
