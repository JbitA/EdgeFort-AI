from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import multiprocessing as mp
import os
from pathlib import Path
import queue as queue_module
import shutil
import time
import zipfile

import numpy as np
import pytest

from edgeai.deploy import ABDeployer
from edgeai.engine import InferenceEngine
from edgeai.errors import DeadlineExceeded, DeploymentError, QueueDropped, QueueFull, ValidationError
from edgeai.health import HealthPolicy
from edgeai.model import LinearModel
from edgeai.registry import Registry
from edgeai.runtime import BoundedInferenceExecutor, OverflowPolicy
from edgeai.telemetry import OfflineSpool
from edgeai.util import InterProcessFileLock
from tests.helpers import signed_registry


def _policy():
    return HealthPolicy(min_accuracy=.9, max_ece=1, max_p95_latency_ms=100, max_rss_mb=5000)


def _xy():
    return np.array([[2, 0], [0, 2], [3, 0], [0, 3]], np.float32), np.array([0, 1, 0, 1])


def _npy_bytes(value) -> bytes:
    buf = io.BytesIO()
    np.save(buf, value, allow_pickle=False)
    return buf.getvalue()


def _lock_probe(path: str, out) -> None:
    lock = InterProcessFileLock(path)
    out.put("started")
    with lock.shared():
        out.put("acquired")


def _append_spool(path: str, start: int, count: int) -> None:
    spool = OfflineSpool(path, 1024 * 1024)
    for value in range(start, start + count):
        assert spool.append({"i": value})


class _BlockingEngine:
    def __init__(self):
        self.started = mp.Event()
        self.release = mp.Event()

    def predict(self, payload):
        if payload == "block":
            self.started.set()
            assert self.release.wait(5)
        return payload


def test_npz_rejects_unexpected_members_before_load(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(
        p,
        W=np.ones((1, 2), np.float32),
        b=np.zeros(1, np.float32),
        abi=np.array(LinearModel.ABI),
        surprise=np.ones(1, np.float32),
    )
    with pytest.raises(ValidationError, match="unexpected or missing"):
        LinearModel.load(p)


def test_npz_rejects_forged_huge_shape_before_allocation(tmp_path):
    huge = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        huge,
        {
            "descr": np.dtype(np.float32).str,
            "fortran_order": False,
            "shape": (1_000_000_000, 2),
        },
    )
    p = tmp_path / "bomb.npz"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("W.npy", huge.getvalue())
        z.writestr("b.npy", _npy_bytes(np.zeros(1, np.float32)))
        z.writestr("abi.npy", _npy_bytes(np.array(LinearModel.ABI)))
    with pytest.raises(ValidationError, match="array shape exceeds|expanded size"):
        LinearModel.load(p, max_uncompressed_bytes=1024 * 1024)


def test_verified_model_handle_survives_path_swap_without_loading_swapped_bytes(tmp_path):
    registry, _, *_ = signed_registry(tmp_path)
    entry, _ = registry.inspect("1.0.0")
    with registry.open_verified_model(entry, expected_version="1.0.0") as (handle, manifest):
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"not-an-npz")
        os.replace(replacement, entry / "model.npz")
        runtime = LinearModel.load(handle)
        assert runtime.predict([[2, 0]]).tolist() == [0]
        assert manifest.version == "1.0.0"
    with pytest.raises(Exception):
        registry.inspect("1.0.0")


def test_state_deletion_after_startup_fails_closed(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(tmp_path / "deploy", registry, _policy())
    x, y = _xy()
    deployer.deploy("1.0.0", x, y)
    deployer.state_path.unlink()
    with pytest.raises(DeploymentError, match="state missing"):
        deployer.state()


def test_recovery_restores_backup_when_slot_swap_crashed_before_state_commit(tmp_path):
    registry, *_ = signed_registry(tmp_path, (("1.0.0", 1), ("1.0.1", 2), ("1.0.2", 3)))
    root = tmp_path / "deploy"
    deployer = ABDeployer(root, registry, _policy())
    x, y = _xy()
    deployer.deploy("1.0.0", x, y)
    deployer.deploy("1.0.1", x, y)
    state = deployer.state()
    assert state.active_slot == "B" and state.slots["A"] == "1.0.0"

    slot_a = deployer._slot_dir("A")
    backup = root / ".backup-A"
    os.replace(slot_a, backup)
    shutil.copytree(registry.inspect("1.0.2")[0], slot_a)

    recovered = ABDeployer(root, registry, _policy())
    assert recovered.state().slots["A"] == "1.0.0"
    assert registry.verify_directory(recovered._slot_dir("A"), expected_version="1.0.0").version == "1.0.0"
    assert not backup.exists()


def test_recovery_discards_stale_backup_after_state_commit(tmp_path):
    registry, *_ = signed_registry(tmp_path, (("1.0.0", 1), ("1.0.1", 2), ("1.0.2", 3)))
    root = tmp_path / "deploy"
    deployer = ABDeployer(root, registry, _policy())
    x, y = _xy()
    deployer.deploy("1.0.0", x, y)
    deployer.deploy("1.0.1", x, y)
    deployer.deploy("1.0.2", x, y)
    backup = root / ".backup-A"
    shutil.copytree(registry.inspect("1.0.0")[0], backup)

    recovered = ABDeployer(root, registry, _policy())
    assert recovered.active_version() == "1.0.2"
    assert not backup.exists()


def test_interprocess_file_lock_blocks_other_process(tmp_path):
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    path = str(tmp_path / "lock")
    lock = InterProcessFileLock(path)
    with lock.exclusive():
        process = ctx.Process(target=_lock_probe, args=(path, out))
        process.start()
        assert out.get(timeout=8) == "started"
        with pytest.raises(queue_module.Empty):
            out.get(timeout=0.15)
    assert out.get(timeout=8) == "acquired"
    process.join(timeout=2)
    assert process.exitcode == 0


def test_telemetry_multi_process_append_preserves_all_records(tmp_path):
    ctx = mp.get_context("spawn")
    path = str(tmp_path / "spool.jsonl")
    p1 = ctx.Process(target=_append_spool, args=(path, 0, 100))
    p2 = ctx.Process(target=_append_spool, args=(path, 100, 100))
    p1.start(); p2.start(); p1.join(timeout=5); p2.join(timeout=5)
    assert p1.exitcode == 0 and p2.exitcode == 0
    spool = OfflineSpool(path, 1024 * 1024)
    events = []
    assert spool.flush(events.append) == 200
    assert {event["i"] for event in events} == set(range(200))


def test_executor_reject_policy_bounds_waiting_work():
    engine = _BlockingEngine()
    executor = BoundedInferenceExecutor(engine, capacity=1, policy=OverflowPolicy.REJECT, workers=1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(executor.submit, "block")
        assert engine.started.wait(2)
        second = pool.submit(executor.submit, "queued")
        deadline = time.monotonic() + 2
        while executor.stats()["depth"] != 1 and time.monotonic() < deadline:
            time.sleep(.005)
        with pytest.raises(QueueFull):
            executor.submit("rejected")
        engine.release.set()
        assert first.result(timeout=2) == "block"
        assert second.result(timeout=2) == "queued"
    executor.close()


def test_executor_drop_oldest_completes_evicted_caller_with_error():
    engine = _BlockingEngine()
    executor = BoundedInferenceExecutor(engine, capacity=1, policy=OverflowPolicy.DROP_OLDEST, workers=1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(executor.submit, "block")
        assert engine.started.wait(2)
        evicted = pool.submit(executor.submit, "old")
        deadline = time.monotonic() + 2
        while executor.stats()["depth"] != 1 and time.monotonic() < deadline:
            time.sleep(.005)
        newest = pool.submit(executor.submit, "new")
        with pytest.raises(QueueDropped):
            evicted.result(timeout=2)
        engine.release.set()
        assert first.result(timeout=2) == "block"
        assert newest.result(timeout=2) == "new"
    executor.close()


def test_executor_queue_deadline_returns_without_waiting_for_blocked_worker():
    engine = _BlockingEngine()
    executor = BoundedInferenceExecutor(engine, capacity=2, policy=OverflowPolicy.REJECT, workers=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.submit, "block")
        assert engine.started.wait(2)
        started = time.monotonic()
        with pytest.raises(DeadlineExceeded):
            executor.submit("expires", deadline_s=.05)
        assert time.monotonic() - started < .5
        engine.release.set()
        assert first.result(timeout=2) == "block"
    executor.close()


class _RaceDeployer:
    def active_identity(self):
        return 1, "1.0.0"

    def active_snapshot(self):
        return LinearModel([[0, 1], [1, 0]], [0, 0]), "1.0.1", 2

    def active_version(self):
        return "1.0.1"

    def state(self):
        class S:
            def to_dict(self):
                return {}
        return S()


def test_engine_reports_version_from_same_runtime_snapshot():
    engine = InferenceEngine(_RaceDeployer(), max_batch=4, max_input_features=2)
    pred, _, version = engine.predict([[2, 0]])
    assert pred.tolist() == [1]
    assert version == "1.0.1"

from edgeai.model_io import ArrayHeader, open_regular_binary, preflight_npz, validate_linear_headers
from edgeai.service import create_app
from edgeai.types import DeploymentState
from fastapi.testclient import TestClient


def test_model_io_rejects_missing_file_and_invalid_preflight_arguments(tmp_path):
    with pytest.raises(ValidationError, match="cannot be opened safely"):
        open_regular_binary(tmp_path / "missing.npz")
    buf = io.BytesIO(b"not-a-zip")
    with pytest.raises(ValueError):
        preflight_npz(buf, {"W"}, max_uncompressed_bytes=True)
    with pytest.raises(ValueError):
        preflight_npz(buf, {"W"}, max_uncompressed_bytes=10)
    with pytest.raises(ValueError):
        preflight_npz(buf, set())
    with pytest.raises(ValidationError, match="invalid NPZ"):
        preflight_npz(buf, {"W"})


def test_model_io_rejects_object_dtype_and_empty_member(tmp_path):
    object_path = tmp_path / "object.npz"
    np.savez_compressed(object_path, W=np.array([object()], dtype=object))
    with object_path.open("rb") as handle:
        with pytest.raises(ValidationError, match="object dtypes"):
            preflight_npz(handle, {"W"})

    empty_path = tmp_path / "empty.npz"
    with zipfile.ZipFile(empty_path, "w") as archive:
        archive.writestr("W.npy", b"")
    with empty_path.open("rb") as handle:
        with pytest.raises(ValidationError, match="empty NPZ member"):
            preflight_npz(handle, {"W"})


def test_model_io_header_schema_rejections():
    f4 = np.dtype(np.float32)
    i1 = np.dtype(np.int8)
    text = np.dtype("<U9")
    good_abi = ArrayHeader((), text, False, text.itemsize)

    with pytest.raises(ValidationError, match="rank-2"):
        validate_linear_headers({"W": ArrayHeader((2,), f4, False, 8), "b": ArrayHeader((2,), f4, False, 8), "abi": good_abi}, quantized=False)
    with pytest.raises(ValidationError, match="bias shape"):
        validate_linear_headers({"W": ArrayHeader((2, 2), f4, False, 16), "b": ArrayHeader((1,), f4, False, 4), "abi": good_abi}, quantized=False)
    with pytest.raises(ValidationError, match="ABI marker must be scalar"):
        validate_linear_headers({"W": ArrayHeader((2, 2), f4, False, 16), "b": ArrayHeader((2,), f4, False, 8), "abi": ArrayHeader((1,), text, False, text.itemsize)}, quantized=False)
    with pytest.raises(ValidationError, match="float model"):
        validate_linear_headers({"W": ArrayHeader((2, 2), i1, False, 4), "b": ArrayHeader((2,), f4, False, 8), "abi": good_abi}, quantized=False)
    with pytest.raises(ValidationError, match="runtime ABI marker must be a string"):
        validate_linear_headers({"W": ArrayHeader((2, 2), f4, False, 16), "b": ArrayHeader((2,), f4, False, 8), "abi": ArrayHeader((), f4, False, 4)}, quantized=False)

    base = {
        "qW": ArrayHeader((2, 2), i1, False, 4),
        "scales": ArrayHeader((2,), f4, False, 8),
        "b": ArrayHeader((2,), f4, False, 8),
        "abi": good_abi,
    }
    bad = dict(base); bad["qW"] = ArrayHeader((2, 2), f4, False, 16)
    with pytest.raises(ValidationError, match="int8"):
        validate_linear_headers(bad, quantized=True)
    bad = dict(base); bad["scales"] = ArrayHeader((1,), f4, False, 4)
    with pytest.raises(ValidationError, match="scale shape"):
        validate_linear_headers(bad, quantized=True)
    bad = dict(base); bad["b"] = ArrayHeader((2,), np.dtype(np.float64), False, 16)
    with pytest.raises(ValidationError, match="scales and bias"):
        validate_linear_headers(bad, quantized=True)


def test_executor_drop_newest_and_close_paths():
    engine = _BlockingEngine()
    executor = BoundedInferenceExecutor(engine, capacity=1, policy=OverflowPolicy.DROP_NEWEST, workers=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.submit, "block")
        assert engine.started.wait(2)
        queued = pool.submit(executor.submit, "queued")
        deadline = time.monotonic() + 2
        while executor.stats()["depth"] != 1 and time.monotonic() < deadline:
            time.sleep(.005)
        with pytest.raises(QueueFull, match="dropped newest"):
            executor.submit("newest")
        executor.close(wait=False)
        with pytest.raises(Exception):
            queued.result(timeout=2)
        engine.release.set()
        assert first.result(timeout=2) == "block"
    executor.close()
    with pytest.raises(Exception):
        executor.submit("after-close")


def test_service_maps_unavailable_model_to_503_and_exports_queue_metrics(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(tmp_path / "empty-deploy", registry, _policy())
    engine = InferenceEngine(deployer, max_batch=4, max_input_features=2)
    with TestClient(create_app(engine, api_key="secret")) as client:
        response = client.post("/v1/predict", headers={"x-api-key": "secret"}, json={"inputs": [[1, 2]]})
        assert response.status_code == 503
        metrics = client.get("/metrics", headers={"x-api-key": "secret"})
        assert metrics.status_code == 200
        assert "inference_queue_capacity" in metrics.json()


def test_service_rejects_nonfinite_queue_deadline(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(tmp_path / "deploy", registry, _policy())
    engine = InferenceEngine(deployer)
    for value in (float("nan"), float("inf"), -1.0, True):
        with pytest.raises(ValueError, match="queue_deadline"):
            create_app(engine, api_key="secret", queue_deadline_s=value)


def test_deployment_state_symlink_rejected(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(tmp_path / "deploy", registry, _policy())
    deployer.state_path.unlink()
    target = tmp_path / "state-target"
    target.write_text("{}")
    deployer.state_path.symlink_to(target)
    with pytest.raises(DeploymentError):
        deployer.state()

import threading
from edgeai.observability import Metrics


class _ServiceBlockingEngine:
    def __init__(self):
        self.metrics = Metrics()
        self.started = threading.Event()
        self.release = threading.Event()

    def predict(self, payload):
        if payload == "block":
            self.started.set()
            assert self.release.wait(5)
        return np.array([0]), np.array([[1.0, 0.0]], np.float32), "1.0.0"

    def readiness(self):
        return {"ready": True, "active_version": "1.0.0", "deployment": {}}


def test_http_overload_is_rejected_at_async_admission_boundary():
    engine = _ServiceBlockingEngine()
    app = create_app(
        engine,
        api_key="secret",
        queue_capacity=1,
        queue_policy=OverflowPolicy.REJECT,
        inference_workers=1,
        queue_deadline_s=1.0,
    )
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        executor = app.state.inference_executor
        first = pool.submit(executor.submit, "block")
        assert engine.started.wait(2)
        queued = pool.submit(executor.submit, "queued")
        deadline = time.monotonic() + 2
        while executor.stats()["depth"] != 1 and time.monotonic() < deadline:
            time.sleep(.005)
        response = client.post(
            "/v1/predict",
            headers={"x-api-key": "secret"},
            json={"inputs": [[1, 2]]},
        )
        assert response.status_code == 429
        assert response.headers["retry-after"] == "1"
        engine.release.set()
        first.result(timeout=2)
        queued.result(timeout=2)


def test_http_queue_deadline_returns_504_without_waiting_for_running_inference():
    engine = _ServiceBlockingEngine()
    app = create_app(
        engine,
        api_key="secret",
        queue_capacity=2,
        inference_workers=1,
        queue_deadline_s=.05,
    )
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        executor = app.state.inference_executor
        first = pool.submit(executor.submit, "block")
        assert engine.started.wait(2)
        started = time.monotonic()
        response = client.post(
            "/v1/predict",
            headers={"x-api-key": "secret"},
            json={"inputs": [[1, 2]]},
        )
        elapsed = time.monotonic() - started
        assert response.status_code == 504
        assert elapsed < .5
        engine.release.set()
        first.result(timeout=2)


def test_unauthorized_prediction_is_rejected_before_body_limit(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(tmp_path / "deploy-auth", registry, _policy())
    engine = InferenceEngine(deployer, max_batch=4, max_input_features=2)
    with TestClient(create_app(engine, api_key="secret", max_request_bytes=1024)) as client:
        response = client.post("/v1/predict", content=b"x" * 4096, headers={"content-type": "application/json"})
        assert response.status_code == 401


def test_recovery_removes_uncommitted_slot_from_crash_before_first_state_commit(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    root = tmp_path / "deploy-orphan"
    deployer = ABDeployer(root, registry, _policy())
    slot_a = deployer._slot_dir("A")
    shutil.copytree(registry.inspect("1.0.0")[0], slot_a)
    recovered = ABDeployer(root, registry, _policy())
    assert recovered.active_version() is None
    assert not slot_a.exists()


def test_recovery_rejects_slot_sequence_above_durable_floor(tmp_path):
    registry, *_ = signed_registry(tmp_path, (("1.0.0", 1), ("1.0.1", 2)))
    root = tmp_path / "deploy-floor"
    deployer = ABDeployer(root, registry, _policy())
    x, y = _xy()
    deployer.deploy("1.0.0", x, y)
    state = deployer.state().to_dict()
    state["slots"][state["active_slot"]] = "1.0.1"
    # Make the slot and state agree on version while deliberately leaving floor=1.
    active_dir = deployer._slot_dir(state["active_slot"])
    shutil.rmtree(active_dir)
    shutil.copytree(registry.inspect("1.0.1")[0], active_dir)
    deployer.state_path.write_text(__import__("json").dumps(state))
    with pytest.raises(DeploymentError, match="exceeds anti-rollback floor"):
        ABDeployer(root, registry, _policy())


def test_slot_must_match_canonical_registry_manifest_even_if_alternate_is_validly_signed(tmp_path):
    from edgeai.artifact import TrustedKeyring, build_manifest, sign_manifest
    from edgeai.types import ModelMetrics

    registry, _, private_key, public_key, key_id = signed_registry(tmp_path)
    root = tmp_path / "deploy-canonical"
    deployer = ABDeployer(root, registry, _policy())
    x, y = _xy()
    deployer.deploy("1.0.0", x, y)

    alternate_model = LinearModel([[0, 1], [1, 0]], [0, 0])
    alternate_path = tmp_path / "alternate.npz"
    alternate_model.save(alternate_path)
    alternate_manifest = build_manifest(
        alternate_path,
        "1.0.0",
        1,
        "float32-linear",
        alternate_model.ABI,
        ModelMetrics(1, 0, 1, 100),
        created_utc="2026-01-01T00:00:00+00:00",
    )
    alternate_registry = Registry(
        tmp_path / "alternate-registry", TrustedKeyring({key_id: public_key})
    )
    alternate_registry.add(
        alternate_path,
        alternate_manifest,
        sign_manifest(alternate_manifest, private_key),
        key_id,
    )

    active_slot = deployer.state().active_slot
    active_dir = deployer._slot_dir(active_slot)
    shutil.rmtree(active_dir)
    shutil.copytree(alternate_registry.inspect("1.0.0")[0], active_dir)

    with pytest.raises(DeploymentError, match="canonical registry release"):
        deployer.active_runtime()


def test_interprocess_lock_rejects_symlink_lockfile(tmp_path):
    target = tmp_path / "target"
    target.write_text("do not follow")
    lock_path = tmp_path / "unsafe.lock"
    lock_path.symlink_to(target)
    with pytest.raises(RuntimeError, match="cannot be opened safely"):
        with InterProcessFileLock(lock_path).exclusive():
            pass


def test_telemetry_spool_rejects_symlink_path(tmp_path):
    target = tmp_path / "target-spool"
    target.write_text("sensitive")
    path = tmp_path / "spool.jsonl"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        OfflineSpool(path, 1024)
    assert target.read_text() == "sensitive"
