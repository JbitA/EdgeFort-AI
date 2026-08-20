from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import resource
import time

import numpy as np

from .errors import HealthGateError
from .metrics import accuracy, expected_calibration_error
from .model import LinearModel
from .onnx_backend import OnnxRuntimeCPUModel
from .quantization import DynamicInt8Model, WeightOnlyInt8Model
from .types import ModelMetrics
from .model_io import DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES


@dataclass(frozen=True)
class HealthPolicy:
    min_accuracy: float = 0.94
    max_accuracy_drop: float = 0.02
    max_ece: float = 0.20
    max_p95_latency_ms: float = 20.0
    max_rss_mb: float = 1024.0
    require_finite_logits: bool = True

    def __post_init__(self):
        vals = (
            self.min_accuracy,
            self.max_accuracy_drop,
            self.max_ece,
            self.max_p95_latency_ms,
            self.max_rss_mb,
        )
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("health policy must be finite")
        if not 0 <= self.min_accuracy <= 1:
            raise ValueError("min_accuracy outside [0,1]")
        if not 0 <= self.max_accuracy_drop <= 1:
            raise ValueError("max_accuracy_drop outside [0,1]")
        if not 0 <= self.max_ece <= 1:
            raise ValueError("max_ece outside [0,1]")
        if self.max_p95_latency_ms <= 0 or self.max_rss_mb <= 0:
            raise ValueError("resource limits must be positive")


_RUNTIME_TYPES = {
    "float32-linear": LinearModel,
    "int8-weight-linear": WeightOnlyInt8Model,
    "int8-dynamic-linear": DynamicInt8Model,
    "onnxruntime-cpu": OnnxRuntimeCPUModel,
}


def expected_runtime_abi(kind: str) -> str:
    runtime_type = _RUNTIME_TYPES.get(kind)
    if runtime_type is None:
        raise HealthGateError(f"unsupported model kind {kind}")
    return runtime_type.ABI


def load_runtime(
    source,
    kind: str,
    runtime_abi: str | None = None,
    *,
    max_uncompressed_bytes: int = DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES,
):
    runtime_type = _RUNTIME_TYPES.get(kind)
    if runtime_type is None:
        raise HealthGateError(f"unsupported model kind {kind}")
    if runtime_abi is not None and runtime_abi != runtime_type.ABI:
        raise HealthGateError(
            f"runtime ABI mismatch for {kind}: signed={runtime_abi!r}, expected={runtime_type.ABI!r}"
        )
    try:
        return runtime_type.load(source, max_uncompressed_bytes=max_uncompressed_bytes)
    except HealthGateError:
        raise
    except Exception as e:
        # Loading a signed candidate is part of the deployment health boundary. Keep
        # parser/allocation failures in the same operational error family.
        raise HealthGateError(f"model load rejected: {e}") from e


def benchmark(runtime, x, y, iterations=40, batch=32) -> ModelMetrics:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    if len(x) == 0:
        raise HealthGateError("empty validation set")
    logits = runtime.logits(x)
    if not np.isfinite(logits).all():
        raise HealthGateError("non-finite logits")
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    lat = []
    for i in range(iterations):
        start = (i * batch) % max(1, len(x) - batch + 1)
        bx = x[start : start + batch]
        t = time.perf_counter()
        runtime.predict(bx)
        lat.append((time.perf_counter() - t) * 1000)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = float(rss / 1024.0)  # Linux KiB; qualification CI is Linux.
    return ModelMetrics(
        accuracy(y, runtime.predict(x)),
        expected_calibration_error(y, probs),
        float(np.percentile(lat, 95)),
        rss_mb,
    )


def enforce(policy: HealthPolicy, candidate: ModelMetrics, baseline: ModelMetrics | None = None) -> None:
    candidate.validate()
    if candidate.accuracy < policy.min_accuracy:
        raise HealthGateError("accuracy below minimum")
    if baseline and baseline.accuracy - candidate.accuracy > policy.max_accuracy_drop:
        raise HealthGateError("accuracy regression")
    if candidate.ece > policy.max_ece:
        raise HealthGateError("calibration error above limit")
    if candidate.p95_latency_ms > policy.max_p95_latency_ms:
        raise HealthGateError("latency above limit")
    if candidate.max_rss_mb > policy.max_rss_mb:
        raise HealthGateError("memory above limit")
