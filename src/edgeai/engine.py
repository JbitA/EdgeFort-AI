from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np

from .deploy import ABDeployer
from .errors import DeploymentError, ValidationError
from .observability import Metrics


@dataclass(frozen=True)
class PreparedInference:
    """Authenticated runtime snapshot plus validated request tensor.

    The snapshot is deliberately explicit so process-isolated execution can receive a
    trusted runtime object without re-opening deployment paths or racing lifecycle state.
    """

    inputs: np.ndarray
    runtime: object
    version: str
    generation: int


@dataclass(frozen=True)
class IsolatedRuntimeSnapshot:
    """Immutable authenticated model material for one deployment generation."""

    model_bytes: bytes
    model_kind: str
    runtime_abi: str
    version: str
    generation: int
    max_uncompressed_bytes: int


@dataclass(frozen=True)
class PreparedIsolatedInference(IsolatedRuntimeSnapshot):
    """Validated request plus immutable authenticated model bytes for a child worker."""

    inputs: np.ndarray


class InferenceEngine:
    def __init__(
        self,
        deployer: ABDeployer,
        max_batch=128,
        max_input_features=4096,
        max_input_values=524_288,
        max_output_classes=4096,
        max_output_values=262_144,
    ):
        for name, value in (
            ("max_batch", max_batch),
            ("max_input_features", max_input_features),
            ("max_input_values", max_input_values),
            ("max_output_classes", max_output_classes),
            ("max_output_values", max_output_values),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.deployer = deployer
        self.max_batch = max_batch
        self.max_input_features = max_input_features
        self.max_input_values = max_input_values
        self.max_output_classes = max_output_classes
        self.max_output_values = max_output_values
        self.metrics = Metrics()
        self._lock = threading.RLock()
        self._runtime = None
        self._version = None
        self._generation = None
        self._isolated_generation = None
        self._isolated_version = None
        self._isolated_artifact = None

    def _active(self):
        generation, version = self.deployer.active_identity()
        if version is None:
            raise DeploymentError("no active model")
        with self._lock:
            if generation != self._generation or version != self._version:
                runtime, snapshot_version, snapshot_generation = self.deployer.active_snapshot()
                self._runtime = runtime
                self._version = snapshot_version
                self._generation = snapshot_generation
            return self._runtime, self._version, self._generation

    def _validated_array(self, x):
        try:
            a = np.asarray(x, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValidationError("input must be a rectangular numeric rank-2 array") from e
        if a.ndim != 2:
            raise ValidationError("input must be rank-2")
        if a.shape[0] < 1 or a.shape[0] > self.max_batch:
            raise ValidationError("batch outside configured bounds")
        if a.shape[1] < 1 or a.shape[1] > self.max_input_features:
            raise ValidationError("feature count outside configured bounds")
        if a.size > self.max_input_values:
            raise ValidationError("input element count outside configured bounds")
        if not np.isfinite(a).all():
            raise ValidationError("input contains non-finite values")
        return a

    def _validated_logits(self, logits, *, batch_size: int) -> np.ndarray:
        """Enforce the runtime-to-API output contract before argmax/serialization.

        A malformed or compromised backend must not turn a small bounded request into an
        unbounded response allocation or an internal 500 via unexpected tensor rank.
        """
        try:
            raw = np.asarray(logits)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValidationError("runtime output must be a numeric rank-2 array") from e
        if raw.ndim != 2:
            raise ValidationError("runtime output must be rank-2")
        if raw.shape[0] != batch_size:
            raise ValidationError("runtime output batch dimension mismatch")
        classes = raw.shape[1]
        if classes < 1 or classes > self.max_output_classes:
            raise ValidationError("runtime output class count outside configured bounds")
        if raw.size > self.max_output_values:
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

    def prepare(self, x) -> PreparedInference:
        """Validate input and capture one generation-consistent authenticated runtime."""
        a = self._validated_array(x)
        runtime, version, generation = self._active()
        return PreparedInference(a, runtime, version, generation)

    def isolated_runtime_snapshot(self) -> IsolatedRuntimeSnapshot:
        """Capture authenticated model bytes without constructing a live parent runtime.

        The bytes are cached only for one deployment generation. A generation change or
        rollback forces a fresh stable-handle authentication of the active slot.
        """
        generation, version = self.deployer.active_identity()
        if version is None:
            raise DeploymentError("no active model")
        with self._lock:
            if (
                generation != self._isolated_generation
                or version != self._isolated_version
                or self._isolated_artifact is None
            ):
                model_bytes, manifest, snapshot_generation = self.deployer.active_artifact_snapshot()
                self._isolated_generation = snapshot_generation
                self._isolated_version = manifest.version
                self._isolated_artifact = (
                    model_bytes,
                    manifest.model_kind,
                    manifest.runtime_abi,
                )
            model_bytes, model_kind, runtime_abi = self._isolated_artifact
            return IsolatedRuntimeSnapshot(
                model_bytes,
                model_kind,
                runtime_abi,
                self._isolated_version,
                self._isolated_generation,
                self.deployer.max_model_uncompressed_bytes,
            )

    def prepare_isolated(self, x) -> PreparedIsolatedInference:
        """Validate input and bind it to one authenticated isolated runtime snapshot."""
        a = self._validated_array(x)
        snapshot = self.isolated_runtime_snapshot()
        return PreparedIsolatedInference(
            snapshot.model_bytes,
            snapshot.model_kind,
            snapshot.runtime_abi,
            snapshot.version,
            snapshot.generation,
            snapshot.max_uncompressed_bytes,
            a,
        )

    def complete(self, prepared, logits):
        """Validate backend output, update success metrics, and form the API result."""
        validated = self._validated_logits(logits, batch_size=len(prepared.inputs))
        pred = validated.argmax(1)
        self.metrics.inc("inference_requests_total")
        self.metrics.inc("inference_examples_total", len(prepared.inputs))
        return pred, validated, prepared.version

    def record_failure(self) -> None:
        self.metrics.inc("inference_errors_total")

    def observe_latency_ms(self, started_perf_counter: float) -> None:
        self.metrics.observe_latency_ms((time.perf_counter() - started_perf_counter) * 1000)

    def predict(self, x):
        prepared = self.prepare(x)
        started = time.perf_counter()
        try:
            logits = prepared.runtime.logits(prepared.inputs)
            return self.complete(prepared, logits)
        except Exception:
            self.record_failure()
            raise
        finally:
            self.observe_latency_ms(started)

    def isolation_limits(self) -> dict[str, int]:
        """Small immutable contract supplied to process-isolated workers."""
        return {
            "max_batch": self.max_batch,
            "max_input_features": self.max_input_features,
            "max_input_values": self.max_input_values,
            "max_output_classes": self.max_output_classes,
            "max_output_values": self.max_output_values,
            "max_model_transfer_bytes": self.deployer.max_model_uncompressed_bytes,
        }

    def readiness(self):
        version = self.deployer.active_version()
        return {
            "ready": version is not None,
            "active_version": version,
            "deployment": self.deployer.state().to_dict(),
        }
