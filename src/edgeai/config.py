from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from .health import HealthPolicy
from .qualification import QualificationLimits
from .runtime import ExecutionMode, OverflowPolicy


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate configuration key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _bounded_int(name: str, value, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return value


def _bounded_number(name: str, value, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"invalid {name}")
    return float(value)


@dataclass(frozen=True)
class RuntimeConfig:
    deployment_root: str
    registry_root: str
    api_key_env: str = "EDGEAI_API_KEY"
    queue_capacity: int = 64
    queue_policy: str = OverflowPolicy.REJECT.value
    max_batch: int = 128
    max_input_features: int = 4096
    max_input_values: int = 524_288
    max_output_classes: int = 4096
    max_output_values: int = 262_144
    max_request_bytes: int = 8 * 1024 * 1024
    inference_workers: int = 1
    queue_deadline_s: float = 1.0
    execution_mode: str = ExecutionMode.THREAD.value
    execution_timeout_s: float | None = None
    process_control_timeout_s: float = 5.0
    process_memory_limit_mb: int | None = None
    process_nofile_limit: int = 64
    process_restart_limit: int = 8
    process_restart_window_s: float = 60.0
    shutdown_timeout_s: float = 5.0
    max_model_uncompressed_bytes: int = 256 * 1024 * 1024
    isolate_health_gate: bool = True
    health_gate_timeout_s: float = 30.0
    health_gate_memory_limit_mb: int | None = None
    max_validation_rows: int = 100_000
    max_validation_features: int = 1_000_000
    max_validation_values: int = 10_000_000
    health_benchmark_iterations: int = 40
    health_benchmark_batch: int = 32
    min_accuracy: float = 0.94
    max_accuracy_drop: float = 0.02
    max_ece: float = 0.20
    max_p95_latency_ms: float = 20.0
    max_rss_mb: float = 1024.0

    def validate(self):
        for name in ("deployment_root", "registry_root", "api_key_env"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid {name}")
        _bounded_int("queue_capacity", self.queue_capacity, 1, 100_000)
        if not isinstance(self.queue_policy, str):
            raise ValueError("invalid queue_policy")
        OverflowPolicy(self.queue_policy)
        _bounded_int("max_batch", self.max_batch, 1, 4096)
        _bounded_int("max_input_features", self.max_input_features, 1, 1_000_000)
        _bounded_int("max_input_values", self.max_input_values, 1, 100_000_000)
        _bounded_int("max_output_classes", self.max_output_classes, 1, 1_000_000)
        _bounded_int("max_output_values", self.max_output_values, 1, 10_000_000)
        _bounded_int("max_request_bytes", self.max_request_bytes, 1024, 256 * 1024 * 1024)
        _bounded_int("inference_workers", self.inference_workers, 1, 64)
        _bounded_number("queue_deadline_s", self.queue_deadline_s, 0, 3600)
        if not isinstance(self.execution_mode, str):
            raise ValueError("invalid execution_mode")
        mode = ExecutionMode(self.execution_mode)
        if self.execution_timeout_s is not None:
            _bounded_number("execution_timeout_s", self.execution_timeout_s, 0.001, 3600)
        if mode is ExecutionMode.PROCESS and self.execution_timeout_s is None:
            raise ValueError("process execution_mode requires execution_timeout_s")
        if mode is ExecutionMode.THREAD and self.execution_timeout_s is not None:
            raise ValueError("execution_timeout_s requires process execution_mode")
        _bounded_number("process_control_timeout_s", self.process_control_timeout_s, 0.001, 3600)
        if self.process_memory_limit_mb is not None:
            _bounded_int("process_memory_limit_mb", self.process_memory_limit_mb, 64, 1_048_576)
        if mode is ExecutionMode.THREAD and self.process_memory_limit_mb is not None:
            raise ValueError("process_memory_limit_mb requires process execution_mode")
        _bounded_int("process_nofile_limit", self.process_nofile_limit, 16, 1_048_576)
        _bounded_int("process_restart_limit", self.process_restart_limit, 1, 10_000)
        _bounded_number("process_restart_window_s", self.process_restart_window_s, 0.001, 86_400)
        _bounded_number("shutdown_timeout_s", self.shutdown_timeout_s, 0, 3600)
        _bounded_int(
            "max_model_uncompressed_bytes",
            self.max_model_uncompressed_bytes,
            1024,
            16 * 1024 * 1024 * 1024,
        )
        if not isinstance(self.isolate_health_gate, bool):
            raise ValueError("invalid isolate_health_gate")
        self.qualification_limits()
        # HealthPolicy owns the semantic ranges for these values; construction also
        # rejects NaN/inf so configuration cannot silently bypass a gate.
        self.health_policy()
        return self

    def qualification_limits(self):
        return QualificationLimits(
            timeout_s=self.health_gate_timeout_s,
            memory_limit_mb=self.health_gate_memory_limit_mb,
            max_validation_rows=self.max_validation_rows,
            max_validation_features=self.max_validation_features,
            max_validation_values=self.max_validation_values,
            iterations=self.health_benchmark_iterations,
            batch=self.health_benchmark_batch,
            nofile_limit=self.process_nofile_limit,
        ).validate()

    def deployer_kwargs(self):
        """Security-relevant ABDeployer options for the configured runtime profile."""
        return {
            "max_model_uncompressed_bytes": self.max_model_uncompressed_bytes,
            "isolate_health_gate": self.isolate_health_gate,
            "qualification_limits": self.qualification_limits(),
        }

    def health_policy(self):
        return HealthPolicy(
            self.min_accuracy,
            self.max_accuracy_drop,
            self.max_ece,
            self.max_p95_latency_ms,
            self.max_rss_mb,
        )


def load_config(path: str | Path) -> RuntimeConfig:
    raw = yaml.load(Path(path).read_text(), Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    allowed = set(RuntimeConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    try:
        return RuntimeConfig(**raw).validate()
    except TypeError as e:
        raise ValueError(f"invalid configuration: {e}") from e
