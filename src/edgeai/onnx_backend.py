from __future__ import annotations

"""Narrow ONNX Runtime CPU backend.

This backend intentionally accepts a small interface contract rather than claiming that
arbitrary ONNX graphs are safe or qualified.  Parsing/session construction is expected to
occur in the existing killable process boundary.  Real ONNX Runtime execution remains an
external qualification item until the optional dependency is exercised on target hardware.
"""

import io
from typing import Any

import numpy as np

from .errors import HealthGateError, ValidationError
from .model_io import DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES, model_source


_MAX_ONNX_DIMENSION = 1_000_000


def _bounded_model_bytes(source, *, max_bytes: int) -> bytes:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1024:
        raise ValueError("max_uncompressed_bytes must be an integer >= 1024")
    with model_source(source) as handle:
        raw = handle.read(max_bytes + 1)
        if not raw:
            raise ValidationError("empty ONNX model artifact")
        if len(raw) > max_bytes:
            raise ValidationError("ONNX model artifact exceeds configured size limit")
        return raw


def _shape(node: Any, *, label: str) -> tuple[object, object]:
    shape = getattr(node, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValidationError(f"ONNX {label} must be rank-2")
    return shape[0], shape[1]


def _static_dim(value, *, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > _MAX_ONNX_DIMENSION
    ):
        raise ValidationError(f"ONNX {label} must be a bounded fixed positive dimension")
    return value


def _batch_dim(value, *, label: str) -> int | None:
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_ONNX_DIMENSION:
        raise ValidationError(f"ONNX {label} batch dimension is invalid")
    return value


class OnnxRuntimeCPUModel:
    """One-input/one-output float32 logits model on CPUExecutionProvider only."""

    ABI = "onnxruntime-cpu-v1"

    def __init__(self, session):
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValidationError("ONNX runtime contract requires exactly one input and one output")
        input_meta, output_meta = inputs[0], outputs[0]
        if getattr(input_meta, "type", None) != "tensor(float)":
            raise ValidationError("ONNX input must be tensor(float)")
        if getattr(output_meta, "type", None) != "tensor(float)":
            raise ValidationError("ONNX output must be tensor(float)")
        input_batch, input_features = _shape(input_meta, label="input")
        output_batch, output_classes = _shape(output_meta, label="output")
        self.input_features = _static_dim(input_features, label="input feature")
        self.output_classes = _static_dim(output_classes, label="output class")
        self.input_batch = _batch_dim(input_batch, label="input")
        self.output_batch = _batch_dim(output_batch, label="output")
        if (
            self.input_batch is not None
            and self.output_batch is not None
            and self.input_batch != self.output_batch
        ):
            raise ValidationError("ONNX fixed input/output batch dimensions disagree")
        input_name = getattr(input_meta, "name", None)
        output_name = getattr(output_meta, "name", None)
        if not isinstance(input_name, str) or not input_name:
            raise ValidationError("ONNX input name is invalid")
        if not isinstance(output_name, str) or not output_name:
            raise ValidationError("ONNX output name is invalid")
        self.input_name = input_name
        self.output_name = output_name
        self._session = session

    @classmethod
    def load(cls, source, *, max_uncompressed_bytes=DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES):
        raw = _bounded_model_bytes(source, max_bytes=max_uncompressed_bytes)
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise HealthGateError(
                "onnxruntime CPU backend requires the optional 'onnx' dependency"
            ) from e

        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            # Avoid CPU-spinning worker pools on thermally constrained edge devices.
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            session = ort.InferenceSession(
                raw,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            providers = list(session.get_providers())
            if providers != ["CPUExecutionProvider"]:
                raise ValidationError("ONNX session did not bind exclusively to CPUExecutionProvider")
            return cls(session)
        except (HealthGateError, ValidationError):
            raise
        except Exception as e:
            raise ValidationError(f"ONNX Runtime session rejected model: {e}") from e

    def logits(self, x):
        try:
            a = np.asarray(x, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValidationError("invalid ONNX input tensor") from e
        if a.ndim != 2 or a.shape[1] != self.input_features:
            raise ValidationError("invalid ONNX input shape")
        if self.input_batch is not None and a.shape[0] != self.input_batch:
            raise ValidationError("input batch does not match fixed ONNX batch dimension")
        if self.output_batch is not None and a.shape[0] != self.output_batch:
            raise ValidationError("request batch does not match fixed ONNX output batch dimension")
        if not np.isfinite(a).all():
            raise ValidationError("non-finite ONNX input")
        try:
            values = self._session.run([self.output_name], {self.input_name: a})
        except Exception as e:
            raise ValidationError(f"ONNX Runtime inference failed: {e}") from e
        if not isinstance(values, (list, tuple)) or len(values) != 1:
            raise ValidationError("ONNX Runtime returned an invalid output collection")
        return values[0]

    def predict(self, x):
        logits = np.asarray(self.logits(x))
        if logits.ndim != 2 or logits.shape[1] != self.output_classes:
            raise ValidationError("ONNX Runtime returned invalid logits shape")
        return logits.argmax(axis=1)
