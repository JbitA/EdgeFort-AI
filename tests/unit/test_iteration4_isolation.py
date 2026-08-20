from __future__ import annotations

import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from edgeai.config import RuntimeConfig
from edgeai.engine import InferenceEngine
from edgeai.errors import BackendProcessError, ExecutionDeadlineExceeded, ValidationError
from edgeai.runtime import BoundedInferenceExecutor, ExecutionMode
from edgeai.service import create_app


class _State:
    def to_dict(self):
        return {"test": True}


class _RuntimeDeployer:
    def __init__(self, runtime):
        self.runtime = runtime
        self.generation = 1
        self.version = "1.0.0"

    def active_identity(self):
        return self.generation, self.version

    def active_snapshot(self):
        return self.runtime, self.version, self.generation

    def active_version(self):
        return self.version

    def state(self):
        return _State()


class _ControllableRuntime:
    def logits(self, x):
        marker = float(x[0, 0])
        if marker == 99.0:
            time.sleep(3)
        if marker == 98.0:
            os._exit(17)
        return np.column_stack((x[:, 0], -x[:, 0])).astype(np.float32)


class _BadShapeRuntime:
    def __init__(self, output):
        self.output = output

    def logits(self, x):
        return self.output


def _engine(runtime, **kwargs):
    return InferenceEngine(
        _RuntimeDeployer(runtime),
        max_batch=4,
        max_input_features=2,
        max_output_classes=8,
        max_output_values=16,
        **kwargs,
    )


def _deployed_engine(tmp_path, *, root_name="deploy"):
    from edgeai.deploy import ABDeployer
    from edgeai.health import HealthPolicy
    from tests.helpers import signed_registry

    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(
        tmp_path / root_name,
        registry,
        HealthPolicy(min_accuracy=.9, max_ece=1, max_p95_latency_ms=100, max_rss_mb=5000),
    )
    x = np.array([[2, 0], [0, 2]], np.float32)
    y = np.array([0, 1])
    deployer.deploy("1.0.0", x, y)
    return InferenceEngine(
        deployer,
        max_batch=4,
        max_input_features=2,
        max_input_values=8,
        max_output_classes=8,
        max_output_values=16,
    )


def test_engine_rejects_unbounded_or_malformed_backend_outputs():
    cases = [
        (np.ones((1, 9), np.float32), "class count"),
        (np.ones((3,), np.float32), "rank-2"),
        (np.ones((2, 2), np.float32), "batch dimension"),
        (np.array([[1.0, np.inf]], np.float32), "non-finite"),
        (np.array([["a", "b"]]), "numeric"),
    ]
    for output, message in cases:
        engine = _engine(_BadShapeRuntime(output))
        with pytest.raises(ValidationError, match=message):
            engine.predict([[1, 0]])
        assert engine.metrics.snapshot()["inference_errors_total"] == 1


def test_engine_enforces_total_output_element_bound():
    engine = InferenceEngine(
        _RuntimeDeployer(_BadShapeRuntime(np.ones((2, 3), np.float32))),
        max_batch=4,
        max_input_features=2,
        max_output_classes=8,
        max_output_values=5,
    )
    with pytest.raises(ValidationError, match="element count"):
        engine.predict([[1, 0], [2, 0]])


def test_process_isolation_hard_timeout_kills_and_restarts_worker(tmp_path):
    engine = _deployed_engine(tmp_path)
    executor = BoundedInferenceExecutor(
        engine,
        capacity=2,
        workers=1,
        execution_mode=ExecutionMode.PROCESS,
        execution_timeout_s=1e-9,
        process_control_timeout_s=10.0,
    )
    try:
        executor.start()
        started = time.monotonic()
        with pytest.raises(ExecutionDeadlineExceeded):
            executor.submit([[2, 0]])
        assert time.monotonic() - started < 1.5

        executor.execution_timeout_s = 1.0
        pred, logits, version = executor.submit([[2, 0]])
        assert pred.tolist() == [0]
        assert logits.shape == (1, 2)
        assert version == "1.0.0"
        stats = executor.stats()
        assert stats["execution_timeouts"] == 1
        assert stats["process_starts"] >= 2
        assert stats["process_restarts"] >= 1
    finally:
        assert executor.close(wait=True, timeout_s=2.0)


def test_process_isolation_contains_external_child_crash_and_recovers(tmp_path):
    engine = _deployed_engine(tmp_path, root_name="deploy-crash")
    executor = BoundedInferenceExecutor(
        engine,
        capacity=2,
        workers=1,
        execution_mode="process",
        execution_timeout_s=1.0,
        process_control_timeout_s=10.0,
    )
    try:
        executor.start()
        runner = executor._process_runners[0]
        process = runner._process
        process.terminate()
        process.join(1)
        pred, _, _ = executor.submit([[3, 0]])
        assert pred.tolist() == [0]
        assert executor.stats()["process_crashes"] >= 1
    finally:
        executor.close(wait=True, timeout_s=2.0)


def test_service_maps_process_execution_timeout_distinctly(tmp_path):
    app = create_app(
        _deployed_engine(tmp_path, root_name="deploy-service-timeout"),
        api_key="secret",
        queue_capacity=2,
        inference_workers=1,
        queue_deadline_s=1.0,
        execution_mode="process",
        execution_timeout_s=1e-9,
        process_control_timeout_s=10.0,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/predict",
            headers={"x-api-key": "secret"},
            json={"inputs": [[2, 0]]},
        )
        assert response.status_code == 504
        assert response.headers["x-edgeai-timeout"] == "execution"
        app.state.inference_executor.execution_timeout_s = 1.0
        ok = client.post(
            "/v1/predict",
            headers={"x-api-key": "secret"},
            json={"inputs": [[2, 0]]},
        )
        assert ok.status_code == 200
        metrics = client.get("/metrics", headers={"x-api-key": "secret"}).json()
        assert metrics["inference_queue_execution_timeouts"] == 1
        assert metrics["inference_queue_execution_mode"] == "process"


def test_execution_mode_configuration_is_fail_closed():
    with pytest.raises(ValueError, match="requires execution_timeout_s"):
        RuntimeConfig("d", "r", execution_mode="process").validate()
    with pytest.raises(ValueError, match="requires process execution_mode"):
        RuntimeConfig("d", "r", execution_timeout_s=1.0).validate()
    with pytest.raises(ValueError):
        RuntimeConfig("d", "r", execution_mode="bogus").validate()
    RuntimeConfig(
        "d",
        "r",
        execution_mode="process",
        execution_timeout_s=1.0,
        process_control_timeout_s=2.0,
        process_memory_limit_mb=2048,
    ).validate()
    with pytest.raises(ValueError, match="requires process execution_mode"):
        RuntimeConfig("d", "r", process_memory_limit_mb=2048).validate()
    for value in (63, 1_048_577, True, 1.5):
        with pytest.raises(ValueError):
            RuntimeConfig(
                "d", "r", execution_mode="process", execution_timeout_s=1.0,
                process_memory_limit_mb=value,
            ).validate()


def test_executor_rejects_invalid_process_timeout_combinations():
    engine = _engine(_ControllableRuntime())
    with pytest.raises(ValueError, match="requires execution_timeout_s"):
        BoundedInferenceExecutor(engine, execution_mode="process")
    with pytest.raises(ValueError, match="requires process execution mode"):
        BoundedInferenceExecutor(engine, execution_timeout_s=1.0)
    with pytest.raises(ValueError, match="control_timeout"):
        BoundedInferenceExecutor(engine, process_control_timeout_s=0)
    with pytest.raises(ValueError, match="requires process execution mode"):
        BoundedInferenceExecutor(engine, process_memory_limit_mb=2048)
    with pytest.raises(ValueError, match="process_memory_limit_mb"):
        BoundedInferenceExecutor(
            engine,
            execution_mode="process",
            execution_timeout_s=1.0,
            process_memory_limit_mb=63,
        )


def test_service_exposes_configured_process_memory_limit(tmp_path):
    app = create_app(
        _deployed_engine(tmp_path, root_name="deploy-memory-limit"),
        api_key="secret",
        execution_mode="process",
        execution_timeout_s=1.0,
        process_control_timeout_s=10.0,
        process_memory_limit_mb=2048,
    )
    with TestClient(app) as client:
        state = client.get("/readyz", headers={"x-api-key": "secret"}).json()[
            "inference_executor"
        ]
        assert state["process_memory_limit_mb"] == 2048
        assert state["process_workers_loaded"] == 1
        assert state["process_ipc_protocol"] == "raw-v1"


def test_process_mode_uses_authenticated_artifact_bootstrap_not_parent_runtime(tmp_path):
    from edgeai.deploy import ABDeployer
    from edgeai.health import HealthPolicy
    from tests.helpers import signed_registry

    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(
        tmp_path / "deploy",
        registry,
        HealthPolicy(min_accuracy=.9, max_ece=1, max_p95_latency_ms=100, max_rss_mb=5000),
    )
    x = np.array([[2, 0], [0, 2]], np.float32)
    y = np.array([0, 1])
    deployer.deploy("1.0.0", x, y)

    engine = InferenceEngine(deployer, max_batch=4, max_input_features=2)
    # The process path should not need to construct the runtime in the web/control process.
    def forbidden_parent_runtime():
        raise AssertionError("parent runtime load should not be used in process mode")
    deployer.active_snapshot = forbidden_parent_runtime

    executor = BoundedInferenceExecutor(
        engine,
        workers=1,
        execution_mode="process",
        execution_timeout_s=1.0,
        process_control_timeout_s=10.0,
    )
    try:
        pred, logits, version = executor.submit([[2, 0]])
        assert pred.tolist() == [0]
        assert logits.shape == (1, 2)
        assert version == "1.0.0"
    finally:
        executor.close(wait=True, timeout_s=2.0)


def test_engine_enforces_total_input_element_bound():
    engine = InferenceEngine(
        _RuntimeDeployer(_ControllableRuntime()),
        max_batch=4,
        max_input_features=4,
        max_input_values=5,
    )
    with pytest.raises(ValidationError, match="input element count"):
        engine.predict([[1, 2, 3], [4, 5, 6]])


def test_readyz_restarts_idle_process_worker_and_reports_executor_state(tmp_path):
    from edgeai.deploy import ABDeployer
    from edgeai.health import HealthPolicy
    from tests.helpers import signed_registry

    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(
        tmp_path / "deploy-ready",
        registry,
        HealthPolicy(min_accuracy=.9, max_ece=1, max_p95_latency_ms=100, max_rss_mb=5000),
    )
    x = np.array([[2, 0], [0, 2]], np.float32)
    y = np.array([0, 1])
    deployer.deploy("1.0.0", x, y)
    engine = InferenceEngine(deployer, max_batch=4, max_input_features=2)
    app = create_app(
        engine,
        api_key="secret",
        execution_mode="process",
        execution_timeout_s=1.0,
        process_control_timeout_s=10.0,
    )
    with TestClient(app) as client:
        executor = app.state.inference_executor
        runner = executor._process_runners[0]
        process = runner._process
        process.terminate(); process.join(1)
        assert not process.is_alive()

        response = client.get("/readyz", headers={"x-api-key": "secret"})
        assert response.status_code == 200
        state = response.json()["inference_executor"]
        assert state["process_workers_alive"] == 1
        assert state["process_restarts"] >= 1


def _run_body_middleware(headers, messages, *, max_bytes=1024):
    import asyncio
    from edgeai.service import RequestBodyLimitMiddleware

    sent = []
    consumed = {"called": False}

    async def app(scope, receive, send):
        consumed["called"] = True
        while True:
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    queue = list(messages)

    async def receive():
        return queue.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": "/v1/predict", "headers": headers}
    asyncio.run(RequestBodyLimitMiddleware(app, max_bytes)(scope, receive, send))
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    return status, consumed["called"]


def test_body_limiter_rejects_ambiguous_http_framing_before_application():
    status, called = _run_body_middleware(
        [(b"content-length", b"2"), (b"content-length", b"2")],
        [{"type": "http.request", "body": b"{}", "more_body": False}],
    )
    assert status == 400 and not called

    status, called = _run_body_middleware(
        [(b"content-length", b"2"), (b"transfer-encoding", b"chunked")],
        [{"type": "http.request", "body": b"{}", "more_body": False}],
    )
    assert status == 400 and not called

    for invalid in (b"-1", b"+2", b" 2", b"2 ", b""):
        status, called = _run_body_middleware(
            [(b"content-length", invalid)],
            [{"type": "http.request", "body": b"{}", "more_body": False}],
        )
        assert status == 400 and not called


def test_body_limiter_checks_declared_length_against_consumed_asgi_body():
    status, called = _run_body_middleware(
        [(b"content-length", b"5")],
        [{"type": "http.request", "body": b"1234", "more_body": False}],
    )
    assert status == 400 and called

    status, called = _run_body_middleware(
        [(b"content-length", b"4")],
        [
            {"type": "http.request", "body": b"12", "more_body": True},
            {"type": "http.request", "body": b"34", "more_body": False},
        ],
    )
    assert status == 204 and called
