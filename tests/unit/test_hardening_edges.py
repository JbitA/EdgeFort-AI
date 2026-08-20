from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from edgeai.artifact import (
    TrustedKeyring,
    build_manifest,
    generate_keypair,
    manifest_from_dict,
    public_key_id,
    sign_manifest,
    verify_artifact,
)
from edgeai.deploy import ABDeployer
from edgeai.engine import InferenceEngine
from edgeai.errors import ArtifactIntegrityError, DeploymentError, HealthGateError, SignatureError, ValidationError
from edgeai.health import HealthPolicy, benchmark, expected_runtime_abi, load_runtime
from edgeai.model import LinearModel
from edgeai.registry import Registry
from edgeai.service import create_app
from edgeai.types import DeploymentState, ModelMetrics
from tests.helpers import signed_registry


def _artifact(tmp_path):
    p = tmp_path / 'model.npz'
    LinearModel([[1, 0], [0, 1]], [0, 0]).save(p)
    return p


def test_manifest_validation_error_edges(tmp_path):
    p = _artifact(tmp_path)
    metrics = ModelMetrics(1, 0, 1, 1)
    with pytest.raises(ValidationError):
        build_manifest(p, '1.0.0', -1, 'float32-linear', LinearModel.ABI, metrics)
    with pytest.raises(ValidationError):
        build_manifest(p, '1.0.0', 1, '', LinearModel.ABI, metrics)
    with pytest.raises(ValidationError):
        build_manifest(tmp_path/'missing', '1.0.0', 1, 'float32-linear', LinearModel.ABI, metrics)
    with pytest.raises(ValidationError):
        build_manifest(p, 'bad', 1, 'float32-linear', LinearModel.ABI, metrics)
    with pytest.raises(ValidationError):
        build_manifest(p, '1.0.0', 1, 'float32-linear', LinearModel.ABI, ModelMetrics(2,0,1,1))


def test_manifest_timestamp_metadata_and_parse_edges(tmp_path):
    p = _artifact(tmp_path)
    m = build_manifest(p, '1.0.0', 1, 'float32-linear', LinearModel.ABI, ModelMetrics(1,0,1,1), created_utc='2026-01-01T00:00:00+00:00')
    for timestamp in ('not-a-date', '2026-01-01T00:00:00', '2026-01-01T03:00:00+03:00'):
        d = m.to_dict(); d['created_utc'] = timestamp
        with pytest.raises(ValidationError): manifest_from_dict(d)
    d = m.to_dict(); d['metadata'] = {'bad': float('nan')}
    with pytest.raises(ValidationError): manifest_from_dict(d)
    with pytest.raises(ValidationError): manifest_from_dict({'schema_version': 1})
    d = m.to_dict(); d['release_sequence'] = -1
    with pytest.raises(ValidationError): manifest_from_dict(d)
    d = m.to_dict(); d['model_kind'] = ''
    with pytest.raises(ValidationError): manifest_from_dict(d)


def test_key_and_artifact_validation_edges(tmp_path):
    p = _artifact(tmp_path)
    m = build_manifest(p, '1.0.0', 1, 'float32-linear', LinearModel.ABI, ModelMetrics(1,0,1,1), created_utc='2026-01-01T00:00:00+00:00')
    with pytest.raises(ValidationError): sign_manifest(m, b'short')
    with pytest.raises(ValidationError): TrustedKeyring({})
    _, pub = generate_keypair()
    with pytest.raises(ValidationError): TrustedKeyring({'': pub})
    with pytest.raises(ValidationError): TrustedKeyring({'bad': b'bad'})
    keyring = TrustedKeyring({public_key_id(pub): pub})
    with pytest.raises(SignatureError): keyring.verify('unknown', m, 'not-a-signature')
    p.unlink()
    with pytest.raises(ArtifactIntegrityError): verify_artifact(p, m)


def test_artifact_size_mismatch_and_symlink_rejected(tmp_path):
    p = _artifact(tmp_path)
    m = build_manifest(p, '1.0.0', 1, 'float32-linear', LinearModel.ABI, ModelMetrics(1,0,1,1), created_utc='2026-01-01T00:00:00+00:00')
    p.write_bytes(b'x')
    with pytest.raises(ArtifactIntegrityError, match='size'):
        verify_artifact(p, m)
    target = tmp_path/'target'; target.write_bytes(b'x')
    link = tmp_path/'link'; link.symlink_to(target)
    with pytest.raises(ArtifactIntegrityError): verify_artifact(link, replace(m, model_bytes=1))


@pytest.mark.parametrize(
    'mutator',
    [
        None,
        lambda s: s.update(extra=1),
        lambda s: s.update(generation=-1),
        lambda s: s.update(highest_release_sequence=-2),
        lambda s: s.update(previous_slot='Z'),
        lambda s: s.update(active_slot='A', previous_slot='A', slots={'A':'1.0.0','B':None}),
        lambda s: s.update(slots={'A':None}),
        lambda s: s.update(active_slot='A', slots={'A':None,'B':None}),
        lambda s: s.update(previous_slot='B', slots={'A':None,'B':None}),
        lambda s: s.update(status='nonsense'),
        lambda s: s.update(last_error=123),
        lambda s: s.update(slots={'A':'../bad','B':None}),
    ],
)
def test_deployment_state_rejects_invalid_shapes(mutator):
    if mutator is None:
        with pytest.raises(DeploymentError): ABDeployer._parse_state([])
        return
    s = DeploymentState().to_dict()
    mutator(s)
    with pytest.raises(DeploymentError): ABDeployer._parse_state(s)


def test_deployer_misc_fail_closed_edges(tmp_path):
    r, *_ = signed_registry(tmp_path)
    root = tmp_path/'deploy'
    d = ABDeployer(root, r)
    assert d.rollback() is False
    with pytest.raises(DeploymentError): d.active_runtime()
    with pytest.raises(DeploymentError): d._slot_dir('Z')
    d.state_path.unlink()
    assert d._read() == DeploymentState()
    # Existing valid state exercises recovery validation on restart.
    d._write(DeploymentState())
    ABDeployer(root, r)


def test_registry_versions_and_corrupt_metadata(tmp_path):
    r, *_ = signed_registry(tmp_path, (('1.0.1', 2), ('1.0.0', 1)))
    assert r.versions() == ['1.0.0', '1.0.1']
    bad = r.root/'1.0.1'/'manifest.json'
    bad.write_text('{bad')
    assert r.versions() == ['1.0.0']
    with pytest.raises(Exception): r.inspect('1.0.1')
    (r.root/'1.0.0'/'signature.txt').unlink()
    with pytest.raises(Exception): r.inspect('1.0.0')


def test_health_loader_and_benchmark_edges(tmp_path):
    assert expected_runtime_abi('float32-linear') == LinearModel.ABI
    with pytest.raises(HealthGateError): expected_runtime_abi('unknown')
    with pytest.raises(HealthGateError): load_runtime(tmp_path/'x', 'unknown')
    class NonFiniteRuntime:
        def logits(self, x): return np.full((len(x),2), np.nan)
        def predict(self, x): return np.zeros(len(x), dtype=int)
    with pytest.raises(HealthGateError): benchmark(NonFiniteRuntime(), np.ones((2,2)), np.array([0,1]))
    with pytest.raises(HealthGateError): benchmark(NonFiniteRuntime(), np.empty((0,2)), np.array([]))
    with pytest.raises(ValueError): HealthPolicy(min_accuracy=float('nan'))


def test_engine_and_service_misc_edges(tmp_path):
    r, *_ = signed_registry(tmp_path)
    d = ABDeployer(tmp_path/'d', r)
    e = InferenceEngine(d, max_batch=2, max_input_features=2)
    with pytest.raises(DeploymentError): e.predict([[1,2]])
    with pytest.raises(ValueError): InferenceEngine(d, max_batch=0)
    c = TestClient(create_app(e, api_key='secret'))
    assert c.get('/healthz').status_code == 200
    assert c.get('/readyz', headers={'x-api-key':'secret'}).status_code == 503
    assert c.get('/metrics', headers={'x-api-key':'secret'}).status_code == 200
    with pytest.raises(ValueError): create_app(e, api_key='secret', max_request_bytes=100)
