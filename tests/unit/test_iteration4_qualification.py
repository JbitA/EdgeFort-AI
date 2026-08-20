from __future__ import annotations

import numpy as np
import pytest

from edgeai.deploy import ABDeployer
from edgeai.errors import HealthGateError
from edgeai.health import HealthPolicy
from edgeai.ipc import (
    IPCProtocolError,
    decode_metrics,
    decode_qualification,
    encode_frame,
    encode_metrics,
    encode_qualification,
)
from edgeai.qualification import QualificationLimits, qualify_model_bytes
from edgeai.types import ModelMetrics
from tests.helpers import signed_registry


def _policy():
    return HealthPolicy(min_accuracy=.9, max_ece=1, max_p95_latency_ms=100, max_rss_mb=5000)


def _artifact(registry, version="1.0.0"):
    directory, _ = registry.inspect(version)
    with registry.open_verified_model(directory, expected_version=version) as (handle, manifest):
        return handle.read(), manifest


def _xy():
    return np.array([[2, 0], [0, 2]], np.float32), np.array([0, 1], np.int64)


def test_qualification_ipc_roundtrip_and_bounds():
    x, y = _xy()
    frame = encode_qualification(
        inputs=x,
        labels=y,
        iterations=3,
        batch=1,
        max_validation_rows=2,
        max_validation_features=2,
        max_validation_values=4,
    )
    got_x, got_y, iterations, batch = decode_qualification(
        frame,
        max_validation_rows=2,
        max_validation_features=2,
        max_validation_values=4,
        max_iterations=3,
    )
    assert got_x.tolist() == x.tolist()
    assert got_y.tolist() == y.tolist()
    assert (iterations, batch) == (3, 1)

    with pytest.raises(IPCProtocolError, match="outside configured bounds"):
        encode_qualification(
            inputs=x,
            labels=y,
            max_validation_rows=1,
            max_validation_features=2,
            max_validation_values=4,
        )
    with pytest.raises(IPCProtocolError, match="labels must be integers"):
        encode_qualification(inputs=x, labels=[0.0, 1.0])
    with pytest.raises(IPCProtocolError, match="shape mismatch"):
        encode_qualification(inputs=x, labels=[0])


def test_qualification_metrics_ipc_is_strict():
    metrics = ModelMetrics(.95, .1, 2.0, 30.0)
    assert decode_metrics(encode_metrics(metrics)) == metrics
    bad = encode_frame(
        "metrics",
        {"accuracy": True, "ece": .1, "max_rss_mb": 1.0, "p95_latency_ms": 1.0},
    )
    with pytest.raises(IPCProtocolError, match="metric field accuracy"):
        decode_metrics(bad)
    with pytest.raises(ValueError, match="finite"):
        encode_metrics(ModelMetrics(float("nan"), .1, 1, 1))


def test_process_qualification_returns_metrics(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    raw, manifest = _artifact(registry)
    x, y = _xy()
    metrics = qualify_model_bytes(
        model_bytes=raw,
        model_kind=manifest.model_kind,
        runtime_abi=manifest.runtime_abi,
        max_uncompressed_bytes=16 * 1024 * 1024,
        x_validation=x,
        y_validation=y,
        limits=QualificationLimits(timeout_s=10, iterations=3, batch=1),
    )
    assert metrics.accuracy == 1.0
    assert metrics.ece <= 1.0
    assert metrics.p95_latency_ms >= 0
    assert metrics.max_rss_mb > 0


def test_process_qualification_has_hard_whole_operation_deadline(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    raw, manifest = _artifact(registry)
    x, y = _xy()
    with pytest.raises(HealthGateError, match="hard wall-clock deadline"):
        qualify_model_bytes(
            model_bytes=raw,
            model_kind=manifest.model_kind,
            runtime_abi=manifest.runtime_abi,
            max_uncompressed_bytes=16 * 1024 * 1024,
            x_validation=x,
            y_validation=y,
            limits=QualificationLimits(timeout_s=1e-9, iterations=3, batch=1),
        )


def test_deployer_isolated_health_gate_does_not_load_runtime_in_lifecycle_process(tmp_path, monkeypatch):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(
        tmp_path / "deploy", registry, _policy(), isolate_health_gate=True
    )
    x, y = _xy()

    def forbidden(*args, **kwargs):
        raise AssertionError("lifecycle process attempted to parse model")

    monkeypatch.setattr("edgeai.deploy.load_runtime", forbidden)
    metrics = deployer.deploy("1.0.0", x, y)
    assert metrics.accuracy == 1.0


def test_deployer_marks_failed_when_isolated_qualification_times_out(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(
        tmp_path / "deploy",
        registry,
        _policy(),
        isolate_health_gate=True,
        qualification_limits=QualificationLimits(timeout_s=1e-9, iterations=3, batch=1),
    )
    x, y = _xy()
    with pytest.raises(HealthGateError, match="hard wall-clock deadline"):
        deployer.deploy("1.0.0", x, y)
    state = deployer.state()
    assert state.status == "failed"
    assert state.active_slot is None


def test_in_process_qualification_remains_explicit_compatibility_path(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(
        tmp_path / "deploy",
        registry,
        _policy(),
        isolate_health_gate=False,
    )
    x, y = _xy()
    assert deployer.deploy("1.0.0", x, y).accuracy == 1.0


def test_isolated_qualification_fails_cleanly_inside_daemon_process(monkeypatch):
    class _Daemon:
        daemon = True

    monkeypatch.setattr("edgeai.qualification.mp.current_process", lambda: _Daemon())
    with pytest.raises(HealthGateError, match="daemon process"):
        qualify_model_bytes(
            model_bytes=b"x",
            model_kind="float32-linear",
            runtime_abi="edgeai.linear.float32.v1",
            max_uncompressed_bytes=1024,
            x_validation=[[1.0]],
            y_validation=[0],
        )


def test_runtime_config_exposes_secure_deployer_profile():
    from edgeai.config import RuntimeConfig

    config = RuntimeConfig("deploy", "registry").validate()
    kwargs = config.deployer_kwargs()
    assert kwargs["isolate_health_gate"] is True
    assert kwargs["max_model_uncompressed_bytes"] == config.max_model_uncompressed_bytes
    assert kwargs["qualification_limits"].timeout_s == config.health_gate_timeout_s


@pytest.mark.parametrize(
    "limits",
    [
        QualificationLimits(timeout_s=0),
        QualificationLimits(memory_limit_mb=63),
        QualificationLimits(max_validation_rows=0),
        QualificationLimits(iterations=0),
    ],
)
def test_qualification_limits_reject_invalid_values(limits):
    with pytest.raises(ValueError):
        limits.validate()
