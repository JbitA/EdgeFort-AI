from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import math

class Slot(str, Enum):
    A = "A"
    B = "B"

class DeploymentStatus(str, Enum):
    EMPTY = "empty"
    STAGED = "staged"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    ece: float
    p95_latency_ms: float
    max_rss_mb: float = 0.0

    def validate(self) -> None:
        values = (self.accuracy, self.ece, self.p95_latency_ms, self.max_rss_mb)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("model metrics must be finite")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("accuracy outside [0,1]")
        if not 0.0 <= self.ece <= 1.0:
            raise ValueError("ece outside [0,1]")
        if self.p95_latency_ms < 0 or self.max_rss_mb < 0:
            raise ValueError("latency/memory must be non-negative")

@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    release_sequence: int
    version: str
    model_kind: str
    runtime_abi: str
    model_sha256: str
    model_bytes: int
    metrics: ModelMetrics
    created_utc: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metrics"] = asdict(self.metrics)
        return d

@dataclass(frozen=True)
class DeploymentState:
    generation: int = 0
    active_slot: str | None = None
    previous_slot: str | None = None
    slots: dict[str, str | None] = field(default_factory=lambda: {"A": None, "B": None})
    highest_release_sequence: int = -1
    status: str = DeploymentStatus.EMPTY.value
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
