from __future__ import annotations

import os

import numpy as np
import pytest

from edgeai.artifact import build_manifest, sign_manifest
from edgeai.deploy import ABDeployer
from edgeai.errors import DeploymentError, RegistryError, RollbackRejected
from edgeai.health import HealthPolicy
from edgeai.model import LinearModel
from edgeai.rollback_anchor import FileRollbackAnchor, MemoryRollbackAnchor
from edgeai.types import ModelMetrics
from tests.helpers import signed_registry


def _policy():
    return HealthPolicy(min_accuracy=.9, max_ece=1, max_p95_latency_ms=100, max_rss_mb=5000)


def _xy():
    return np.array([[2, 0], [0, 2], [3, 0], [0, 3]], np.float32), np.array([0, 1, 0, 1])


def test_external_anchor_advances_and_persists_across_deployer_restart(tmp_path):
    registry, *_ = signed_registry(tmp_path, (("1.0.0", 1), ("1.0.1", 2)))
    anchor = MemoryRollbackAnchor()
    root = tmp_path / "deploy"
    deployer = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    x, y = _xy()

    deployer.deploy("1.0.0", x, y)
    deployer.deploy("1.0.1", x, y)

    assert anchor.read_floor() == 2
    restarted = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    assert restarted.active_version() == "1.0.1"
    assert restarted.state().highest_release_sequence == 2


def test_external_anchor_detects_whole_state_rewind(tmp_path):
    registry, *_ = signed_registry(tmp_path, (("1.0.0", 1), ("1.0.1", 2)))
    anchor = MemoryRollbackAnchor()
    root = tmp_path / "deploy"
    deployer = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    x, y = _xy()

    deployer.deploy("1.0.0", x, y)
    old_state = deployer.state_path.read_bytes()
    deployer.deploy("1.0.1", x, y)
    deployer.state_path.write_bytes(old_state)

    with pytest.raises(DeploymentError, match="ahead.*without recovery journal"):
        ABDeployer(root, registry, _policy(), rollback_anchor=anchor)


def test_recovery_completes_commit_if_anchor_advanced_before_state_write(tmp_path, monkeypatch):
    registry, *_ = signed_registry(tmp_path)
    anchor = MemoryRollbackAnchor()
    root = tmp_path / "deploy"
    deployer = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    x, y = _xy()
    real_write = deployer._write

    def crash_on_committed_state(state):
        if state.highest_release_sequence == 1:
            raise OSError("simulated power loss before state commit")
        return real_write(state)

    monkeypatch.setattr(deployer, "_write", crash_on_committed_state)
    with pytest.raises(OSError, match="power loss"):
        deployer.deploy("1.0.0", x, y)

    assert anchor.read_floor() == 1
    assert deployer.pending_anchor_path.exists()
    recovered = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    assert recovered.active_version() == "1.0.0"
    assert recovered.state().highest_release_sequence == 1
    assert not recovered.pending_anchor_path.exists()


def test_recovery_discards_pre_anchor_transaction_when_anchor_did_not_advance(tmp_path):
    registry, *_ = signed_registry(tmp_path)

    class FailingAnchor(MemoryRollbackAnchor):
        def advance_to(self, release_sequence: int) -> int:
            raise OSError("anchor unavailable")

    anchor = FailingAnchor()
    root = tmp_path / "deploy"
    deployer = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    x, y = _xy()
    with pytest.raises(DeploymentError, match="anchor advance failed"):
        deployer.deploy("1.0.0", x, y)

    recovered = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    assert recovered.active_version() is None
    assert recovered.state().highest_release_sequence == -1
    assert not recovered.pending_anchor_path.exists()
    assert not recovered._slot_dir("A").exists()


def test_external_anchor_makes_allow_downgrade_non_bypassable(tmp_path):
    registry, *_ = signed_registry(tmp_path, (("1.0.0", 1), ("1.0.1", 2)))
    anchor = MemoryRollbackAnchor()
    deployer = ABDeployer(tmp_path / "deploy", registry, _policy(), rollback_anchor=anchor)
    x, y = _xy()
    deployer.deploy("1.0.1", x, y)

    with pytest.raises(RollbackRejected, match="cannot be bypassed"):
        deployer.deploy("1.0.0", x, y, allow_downgrade=True)


def test_registry_rejects_release_sequence_reuse(tmp_path):
    registry, _, private, _, key_id = signed_registry(tmp_path)
    model = LinearModel([[1, 0], [0, 1]], [0, 0])
    path = tmp_path / "second.npz"
    model.save(path)
    manifest = build_manifest(
        path,
        "1.0.1",
        1,
        "float32-linear",
        model.ABI,
        ModelMetrics(1, 0, 1, 100),
        created_utc="2026-01-01T00:00:00+00:00",
    )
    signature = sign_manifest(manifest, private)

    with pytest.raises(RegistryError, match="already assigned"):
        registry.add(path, manifest, signature, key_id)


def test_file_anchor_is_persistent_monotonic_and_symlink_safe(tmp_path):
    path = tmp_path / "anchor" / "floor.json"
    anchor = FileRollbackAnchor(path)
    assert anchor.read_floor() == -1
    assert anchor.advance_to(4) == 4
    assert FileRollbackAnchor(path).read_floor() == 4
    with pytest.raises(DeploymentError, match="cannot decrease"):
        anchor.advance_to(3)

    path.unlink()
    os.symlink(tmp_path / "victim", path)
    with pytest.raises((ValueError, DeploymentError)):
        FileRollbackAnchor(path)


def test_manifest_parser_rejects_type_coercion_unknown_fields_and_bool_numbers(tmp_path):
    from edgeai.artifact import manifest_from_dict
    from edgeai.errors import ValidationError

    registry, *_ = signed_registry(tmp_path)
    manifest = registry.inspect("1.0.0")[1]

    data = manifest.to_dict()
    data["release_sequence"] = "1"
    with pytest.raises(ValidationError, match="release_sequence"):
        manifest_from_dict(data)

    data = manifest.to_dict()
    data["model_bytes"] = True
    with pytest.raises(ValidationError, match="model_bytes"):
        manifest_from_dict(data)

    data = manifest.to_dict()
    data["metrics"]["accuracy"] = True
    with pytest.raises(ValidationError, match="metrics.accuracy"):
        manifest_from_dict(data)

    data = manifest.to_dict()
    data["unsigned_hint"] = "ignored-by-old-parser"
    with pytest.raises(ValidationError, match="fields mismatch"):
        manifest_from_dict(data)


def test_registry_rejects_duplicate_manifest_json_keys_even_if_semantics_match(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    path = registry.root / "1.0.0" / "manifest.json"
    raw = path.read_text()
    assert raw.startswith("{")
    path.write_text('{"schema_version":1,' + raw[1:])

    with pytest.raises(Exception, match="duplicate JSON object key"):
        registry.inspect("1.0.0")


def test_build_manifest_rejects_boolean_and_out_of_range_release_sequences(tmp_path):
    from edgeai.artifact import build_manifest
    from edgeai.errors import ValidationError

    model = LinearModel([[1, 0], [0, 1]], [0, 0])
    path = tmp_path / "model.npz"
    model.save(path)
    metrics = ModelMetrics(1, 0, 1, 100)
    with pytest.raises(ValidationError, match="supported range"):
        build_manifest(path, "1.0.0", True, "float32-linear", model.ABI, metrics)
    with pytest.raises(ValidationError, match="supported range"):
        build_manifest(path, "1.0.0", 1 << 63, "float32-linear", model.ABI, metrics)


def test_telemetry_record_ids_survive_failed_delivery_and_support_idempotent_sink(tmp_path):
    from edgeai.telemetry import OfflineSpool

    spool = OfflineSpool(tmp_path / "spool.jsonl", 4096)
    assert spool.append({"kind": "measurement", "value": 7})
    first = []

    def fail_after_observing(record):
        first.append(record)
        raise RuntimeError("ack lost")

    assert spool.flush_records(fail_after_observing) == 0
    assert len(first) == 1
    assert first[0]["persistent_id"] is True
    assert len(first[0]["event_id"]) == 32

    retry = []
    assert spool.flush_records(retry.append) == 1
    assert retry[0]["event_id"] == first[0]["event_id"]
    assert retry[0]["event"] == {"kind": "measurement", "value": 7}


def test_telemetry_caller_event_id_makes_enqueue_idempotent(tmp_path):
    from edgeai.telemetry import OfflineSpool

    spool = OfflineSpool(tmp_path / "spool.jsonl", 4096)
    event_id = "ab" * 16
    assert spool.append({"i": 1}, event_id=event_id)
    assert spool.append({"i": 1}, event_id=event_id)
    records = []
    assert spool.flush_records(records.append) == 1
    assert records[0]["event_id"] == event_id

    assert spool.append({"i": 1}, event_id=event_id)
    with pytest.raises(ValueError, match="different content"):
        spool.append({"i": 2}, event_id=event_id)


def test_telemetry_legacy_records_remain_readable_but_are_marked_without_persistent_id(tmp_path):
    from edgeai.telemetry import OfflineSpool

    path = tmp_path / "spool.jsonl"
    path.write_text('{"legacy":true}\n')
    spool = OfflineSpool(path, 4096)
    records = []
    assert spool.flush_records(records.append) == 1
    assert records == [
        {"event_id": None, "event": {"legacy": True}, "persistent_id": False}
    ]


def test_executor_shutdown_wait_is_bounded_for_hung_running_inference():
    import threading
    import time
    from edgeai.runtime import BoundedInferenceExecutor

    class HungEngine:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def predict(self, payload):
            self.started.set()
            self.release.wait(5)
            return payload

    engine = HungEngine()
    executor = BoundedInferenceExecutor(engine, capacity=1, workers=1)
    executor.start()
    task = threading.Thread(target=lambda: executor.submit("work"), daemon=True)
    task.start()
    assert engine.started.wait(1)
    started = time.monotonic()
    assert executor.close(wait=True, timeout_s=.05) is False
    assert time.monotonic() - started < .5
    engine.release.set()
    task.join(1)
    assert executor.close(wait=True, timeout_s=1) is True


def test_service_rejects_invalid_shutdown_timeout(tmp_path):
    from edgeai.engine import InferenceEngine
    from edgeai.service import create_app

    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(tmp_path / "deploy", registry, _policy())
    engine = InferenceEngine(deployer, max_batch=4, max_input_features=2)
    with pytest.raises(ValueError, match="shutdown_timeout_s"):
        create_app(engine, api_key="secret", shutdown_timeout_s=float("inf"))


def test_runtime_config_rejects_boolean_numeric_fields_and_nonfinite_shutdown(tmp_path):
    from edgeai.config import RuntimeConfig, load_config

    with pytest.raises(ValueError, match="queue_capacity"):
        RuntimeConfig("d", "r", queue_capacity=True).validate()
    with pytest.raises(ValueError, match="shutdown_timeout_s"):
        RuntimeConfig("d", "r", shutdown_timeout_s=float("nan")).validate()
    path = tmp_path / "config.yaml"
    path.write_text("deployment_root: d\nregistry_root: r\ninference_workers: true\n")
    with pytest.raises(ValueError, match="inference_workers"):
        load_config(path)


def test_runtime_config_rejects_duplicate_yaml_keys(tmp_path):
    from edgeai.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text(
        "deployment_root: d\nregistry_root: r\nqueue_capacity: 8\nqueue_capacity: 64\n"
    )
    with pytest.raises(ValueError, match="duplicate configuration key"):
        load_config(path)



def test_ambiguous_anchor_advance_poison_requires_fresh_recovery(tmp_path):
    registry, *_ = signed_registry(tmp_path)

    class AckLostAnchor(MemoryRollbackAnchor):
        def advance_to(self, release_sequence: int) -> int:
            super().advance_to(release_sequence)
            raise OSError("ack lost after durable advance")

    anchor = AckLostAnchor()
    root = tmp_path / "deploy"
    deployer = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    x, y = _xy()

    with pytest.raises(DeploymentError, match="anchor advance failed"):
        deployer.deploy("1.0.0", x, y)
    assert anchor.read_floor() == 1
    with pytest.raises(DeploymentError, match="transition requires recovery"):
        deployer.state()

    recovered = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    assert recovered.active_version() == "1.0.0"
    assert recovered.state().highest_release_sequence == 1


def test_live_peer_refreshes_cached_anchor_after_successful_deployment(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    anchor = MemoryRollbackAnchor()
    root = tmp_path / "deploy"
    writer = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    reader = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    assert reader._trusted_floor == -1
    x, y = _xy()

    writer.deploy("1.0.0", x, y)

    state = reader.state()
    assert state.highest_release_sequence == 1
    assert reader._trusted_floor == 1


def test_live_peer_fails_closed_when_anchor_advanced_but_state_commit_is_pending(tmp_path, monkeypatch):
    registry, *_ = signed_registry(tmp_path)
    anchor = MemoryRollbackAnchor()
    root = tmp_path / "deploy"
    writer = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    reader = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    x, y = _xy()
    real_write = writer._write

    def crash_before_state_commit(state):
        if state.highest_release_sequence == 1:
            raise OSError("simulated state persistence failure")
        return real_write(state)

    monkeypatch.setattr(writer, "_write", crash_before_state_commit)
    with pytest.raises(OSError, match="state persistence failure"):
        writer.deploy("1.0.0", x, y)

    with pytest.raises(DeploymentError, match="differs from external rollback anchor"):
        reader.state()

    recovered = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    assert recovered.active_version() == "1.0.0"


def test_manifest_validation_rejects_wrong_metrics_container(tmp_path):
    from dataclasses import replace
    from edgeai.errors import ValidationError

    registry, *_ = signed_registry(tmp_path)
    manifest = registry.inspect("1.0.0")[1]
    malformed = replace(manifest, metrics={"accuracy": 1.0})
    with pytest.raises(ValidationError, match="metrics must be a ModelMetrics"):
        sign_manifest(malformed, b"0" * 32)
