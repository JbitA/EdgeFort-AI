from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from edgeai.artifact import build_manifest, manifest_from_dict, sign_manifest
from edgeai.config import RuntimeConfig, load_config
from edgeai.deploy import ABDeployer
from edgeai.errors import DeploymentError, RegistryError, ValidationError
from edgeai.health import HealthPolicy
from edgeai.model import LinearModel
from edgeai.registry import Registry
from edgeai.rollback_anchor import FileRollbackAnchor, MemoryRollbackAnchor
from edgeai.telemetry import OfflineSpool
from edgeai.types import ModelMetrics
from tests.helpers import signed_registry


def _policy():
    return HealthPolicy(min_accuracy=.9, max_ece=1, max_p95_latency_ms=100, max_rss_mb=5000)


def _xy():
    return np.array([[2, 0], [0, 2], [3, 0], [0, 3]], np.float32), np.array([0, 1, 0, 1])


@pytest.mark.parametrize("value", [True, -2, "1", 1.5])
def test_memory_anchor_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        MemoryRollbackAnchor(value)


def test_deployer_wraps_anchor_read_failures_and_invalid_floor(tmp_path):
    registry, *_ = signed_registry(tmp_path)

    class Broken:
        def read_floor(self):
            raise OSError("device unavailable")
        def advance_to(self, release_sequence):
            return release_sequence

    with pytest.raises(DeploymentError, match="anchor read failed"):
        ABDeployer(tmp_path / "d1", registry, _policy(), rollback_anchor=Broken())

    class Invalid(Broken):
        def read_floor(self):
            return True

    with pytest.raises(DeploymentError, match="invalid floor"):
        ABDeployer(tmp_path / "d2", registry, _policy(), rollback_anchor=Invalid())


def test_deployer_rejects_anchor_that_advances_to_wrong_value(tmp_path):
    registry, *_ = signed_registry(tmp_path)

    class WrongAdvance(MemoryRollbackAnchor):
        def advance_to(self, release_sequence):
            super().advance_to(release_sequence)
            return release_sequence + 1

    deployer = ABDeployer(
        tmp_path / "deploy", registry, _policy(), rollback_anchor=WrongAdvance()
    )
    with pytest.raises(DeploymentError, match="unexpected release sequence"):
        deployer.deploy("1.0.0", *_xy())


def test_unconfigured_anchor_helpers_and_state_mismatch_fail_closed(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(tmp_path / "deploy", registry, _policy())
    with pytest.raises(DeploymentError, match="not configured"):
        deployer._read_anchor_floor()
    assert deployer._advance_anchor(3) == 3

    anchored = ABDeployer(
        tmp_path / "anchored", registry, _policy(), rollback_anchor=MemoryRollbackAnchor()
    )
    state = replace(anchored.state(), highest_release_sequence=0)
    with pytest.raises(DeploymentError, match="differs"):
        anchored._validate_state_anchor(state)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: [],
        lambda d: {**d, "schema_version": 2},
        lambda d: {**d, "previous_generation": True},
        lambda d: {**d, "previous_floor": -2},
        lambda d: {**d, "target_slot": "Z"},
        lambda d: {**d, "version": "../bad"},
        lambda d: {**d, "release_sequence": True},
        lambda d: {**d, "release_sequence": -1},
        lambda d: {**d, "previous_floor": 0, "release_sequence": 0},
    ],
)
def test_pending_anchor_schema_rejects_malformed_records(mutation):
    valid = {
        "schema_version": 1,
        "previous_generation": 0,
        "previous_floor": -1,
        "target_slot": "A",
        "version": "1.0.0",
        "release_sequence": 1,
    }
    with pytest.raises(DeploymentError):
        ABDeployer._parse_pending_anchor(mutation(valid))


def test_pending_anchor_without_configured_anchor_fails_closed(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    root = tmp_path / "deploy"
    deployer = ABDeployer(root, registry, _policy())
    pending = {
        "schema_version": 1,
        "previous_generation": 0,
        "previous_floor": -1,
        "target_slot": "A",
        "version": "1.0.0",
        "release_sequence": 1,
    }
    deployer.pending_anchor_path.write_text(json.dumps(pending))
    with pytest.raises(DeploymentError, match="no rollback anchor"):
        ABDeployer(root, registry, _policy())


def test_state_ahead_of_external_anchor_fails_closed(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    root = tmp_path / "deploy"
    deployer = ABDeployer(root, registry, _policy())
    deployer.deploy("1.0.0", *_xy())
    with pytest.raises(DeploymentError, match="exceeds external"):
        ABDeployer(root, registry, _policy(), rollback_anchor=MemoryRollbackAnchor(0))


def test_recovery_removes_stale_post_commit_anchor_journal(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    root = tmp_path / "deploy"
    anchor = MemoryRollbackAnchor()
    deployer = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    deployer.deploy("1.0.0", *_xy())
    pending = {
        "schema_version": 1,
        "previous_generation": 0,
        "previous_floor": -1,
        "target_slot": "A",
        "version": "1.0.0",
        "release_sequence": 1,
    }
    deployer.pending_anchor_path.write_text(json.dumps(pending))
    restarted = ABDeployer(root, registry, _policy(), rollback_anchor=anchor)
    assert restarted.active_version() == "1.0.0"
    assert not restarted.pending_anchor_path.exists()


def test_file_anchor_rejects_corrupt_oversized_and_wrong_schema(tmp_path):
    path = tmp_path / "floor.json"
    anchor = FileRollbackAnchor(path)

    path.write_text("{bad")
    with pytest.raises(DeploymentError, match="invalid rollback anchor"):
        anchor.read_floor()

    path.write_bytes(b"x" * 5000)
    with pytest.raises(DeploymentError, match="size limit"):
        anchor.read_floor()

    path.write_text('{"floor":-1,"schema_version":2}')
    with pytest.raises(DeploymentError, match="unsupported"):
        anchor.read_floor()

    path.write_text('{"floor":true,"schema_version":1}')
    with pytest.raises(DeploymentError, match="integer"):
        anchor.read_floor()

    path.write_text('{"floor":-1,"schema_version":1,"extra":0}')
    with pytest.raises(DeploymentError, match="structure"):
        anchor.read_floor()


def test_file_anchor_same_value_is_noop_and_missing_file_fails(tmp_path):
    path = tmp_path / "floor.json"
    anchor = FileRollbackAnchor(path, initial=2)
    before = path.read_bytes()
    assert anchor.advance_to(2) == 2
    assert path.read_bytes() == before
    path.unlink()
    with pytest.raises(DeploymentError, match="opened safely"):
        anchor.read_floor()


def test_manifest_strict_schema_additional_edges(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    manifest = registry.inspect("1.0.0")[1]

    cases = []
    d = manifest.to_dict(); d["schema_version"] = True; cases.append(d)
    d = manifest.to_dict(); d["version"] = 1; cases.append(d)
    d = manifest.to_dict(); d["metadata"] = []; cases.append(d)
    d = manifest.to_dict(); d["metrics"]["ece"] = "0"; cases.append(d)
    d = manifest.to_dict(); d["metrics"].pop("ece"); cases.append(d)
    for data in cases:
        with pytest.raises(ValidationError):
            manifest_from_dict(data)
    with pytest.raises(ValidationError):
        manifest_from_dict([])


def test_registry_uniqueness_fails_closed_on_unexpected_or_corrupt_published_entries(tmp_path):
    registry, _, private, _, key_id = signed_registry(tmp_path)
    model = LinearModel([[1, 0], [0, 1]], [0, 0])
    path = tmp_path / "new.npz"; model.save(path)
    manifest = build_manifest(
        path, "1.0.1", 2, "float32-linear", model.ABI,
        ModelMetrics(1, 0, 1, 100), created_utc="2026-01-01T00:00:00+00:00"
    )
    signature = sign_manifest(manifest, private)

    (registry.root / "junk").write_text("not a directory")
    with pytest.raises(RegistryError, match="non-directory"):
        registry.add(path, manifest, signature, key_id)
    (registry.root / "junk").unlink()

    (registry.root / "1.0.0" / "manifest.json").write_text("{bad")
    with pytest.raises(RegistryError, match="cannot establish"):
        registry.add(path, manifest, signature, key_id)


def test_registry_release_identity_rejects_wrong_sequence(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    with pytest.raises(RegistryError, match="expected release sequence"):
        registry.assert_release_sequence_identity("1.0.0", 99)


def test_telemetry_rejects_bad_ids_and_reports_legacy_and_persistent_counts(tmp_path):
    path = tmp_path / "spool.jsonl"
    path.write_text('{"legacy":1}\n')
    spool = OfflineSpool(path, 4096)
    with pytest.raises(ValueError, match="event_id"):
        spool.append({"i": 1}, event_id="BAD")
    assert spool.append({"i": 2}, event_id="cd" * 16)
    stats = spool.stats()
    assert stats["legacy_records"] == 1
    assert stats["persistent_id_records"] == 1


def test_telemetry_duplicate_scan_ignores_corrupt_legacy_line(tmp_path):
    path = tmp_path / "spool.jsonl"
    path.write_bytes(b"{bad}\n")
    spool = OfflineSpool(path, 4096)
    assert spool.append({"i": 1}, event_id="ef" * 16)


def test_telemetry_detects_external_spool_growth(tmp_path):
    path = tmp_path / "spool.jsonl"
    spool = OfflineSpool(path, 1024)
    path.write_bytes(b"x" * 1025)
    with pytest.raises(ValueError, match="exceeds"):
        spool.stats()
    with pytest.raises(ValueError, match="exceeds"):
        spool.flush(lambda _: None)
    with pytest.raises(ValueError, match="exceeds"):
        spool.append({"i": 1})


def test_config_strict_type_and_shape_edges(tmp_path):
    with pytest.raises(ValueError, match="deployment_root"):
        RuntimeConfig("", "r").validate()
    with pytest.raises(ValueError, match="queue_policy"):
        RuntimeConfig("d", "r", queue_policy=1).validate()
    with pytest.raises(ValueError, match="queue_deadline_s"):
        RuntimeConfig("d", "r", queue_deadline_s=True).validate()

    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
    path.write_text("deployment_root: d\n")
    with pytest.raises(ValueError, match="invalid configuration"):
        load_config(path)



def test_release_sequence_range_is_consistent_across_state_journal_and_anchor(tmp_path):
    from edgeai.deploy import ABDeployer
    from edgeai.errors import DeploymentError
    from edgeai.rollback_anchor import MemoryRollbackAnchor
    from edgeai.util import MAX_RELEASE_SEQUENCE

    registry, *_ = signed_registry(tmp_path)
    with pytest.raises(ValueError, match="rollback anchor value"):
        MemoryRollbackAnchor(MAX_RELEASE_SEQUENCE + 1)

    deployer = ABDeployer(tmp_path / "deploy", registry, _policy())
    state = deployer.state().to_dict()
    state["highest_release_sequence"] = MAX_RELEASE_SEQUENCE + 1
    with pytest.raises(DeploymentError, match="anti-rollback floor"):
        deployer._parse_state(state)

    pending = {
        "schema_version": 1,
        "previous_generation": 0,
        "previous_floor": -1,
        "target_slot": "A",
        "version": "1.0.0",
        "release_sequence": MAX_RELEASE_SEQUENCE + 1,
    }
    with pytest.raises(DeploymentError, match="pending anchor release sequence"):
        deployer._parse_pending_anchor(pending)
