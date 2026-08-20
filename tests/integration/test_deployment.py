import numpy as np,pytest,json
from tests.helpers import signed_registry
from edgeai.deploy import ABDeployer
from edgeai.health import HealthPolicy
from edgeai.errors import RollbackRejected,HealthGateError
from edgeai.artifact import *
from edgeai.quantization import quantize_dynamic
from edgeai.model import LinearModel
from edgeai.types import ModelMetrics

def xy():return np.array([[2,0],[0,2],[3,0],[0,3]],np.float32),np.array([0,1,0,1])
def test_deploy_two_slots_and_rollback(tmp_path):
 r,_,priv,pub,kid=signed_registry(tmp_path,(('1.0.0',1),('1.0.1',2)));d=ABDeployer(tmp_path/'deploy',r,HealthPolicy(min_accuracy=.9,max_ece=1,max_p95_latency_ms=100,max_rss_mb=5000));x,y=xy()
 d.deploy('1.0.0',x,y);assert d.active_version()=='1.0.0';d.deploy('1.0.1',x,y);assert d.active_version()=='1.0.1';assert d.rollback();assert d.active_version()=='1.0.0';assert d.state().highest_release_sequence==2
def test_anti_rollback_floor_survives_rollback(tmp_path):
 r,*_=signed_registry(tmp_path,(('1.0.0',1),('1.0.1',2)));d=ABDeployer(tmp_path/'deploy',r,HealthPolicy(min_accuracy=.9,max_ece=1,max_p95_latency_ms=100,max_rss_mb=5000));x,y=xy();d.deploy('1.0.0',x,y);d.deploy('1.0.1',x,y);d.rollback()
 with pytest.raises(RollbackRejected):d.deploy('1.0.0',x,y)
def test_failed_health_keeps_active(tmp_path):
 r,_,priv,pub,kid=signed_registry(tmp_path,(('1.0.0',1),));d=ABDeployer(tmp_path/'deploy',r,HealthPolicy(min_accuracy=.9,max_ece=1,max_p95_latency_ms=100,max_rss_mb=5000));x,y=xy();d.deploy('1.0.0',x,y)
 # add bad all-class-zero model
 m=LinearModel([[0,0],[0,0]],[1,0]);p=tmp_path/'bad.npz';m.save(p);man=build_manifest(p,'1.0.1',2,'float32-linear',m.ABI,ModelMetrics(.5,.5,1,100),created_utc='2026-01-01T00:00:00+00:00');sig=sign_manifest(man,priv);r.add(p,man,sig,kid)
 with pytest.raises(HealthGateError):d.deploy('1.0.1',x,y)
 assert d.active_version()=='1.0.0' and d.state().status=='failed'
def test_crash_recovery_removes_candidate(tmp_path):
 r,*_=signed_registry(tmp_path);root=tmp_path/'deploy';root.mkdir();(root/'.candidate-A').mkdir();ABDeployer(root,r);assert not (root/'.candidate-A').exists()
def test_state_corruption_fails_closed(tmp_path):
 r,*_=signed_registry(tmp_path);d=ABDeployer(tmp_path/'deploy',r);d.state_path.write_text('{bad')
 with pytest.raises(Exception):d.state()

def test_active_slot_tamper_rejected_before_execution(tmp_path):
    r, *_ = signed_registry(tmp_path)
    d = ABDeployer(tmp_path/'deploy', r, HealthPolicy(min_accuracy=.9,max_ece=1,max_p95_latency_ms=100,max_rss_mb=5000))
    x, y = xy()
    d.deploy('1.0.0', x, y)
    slot = d._slot_dir(d.state().active_slot)
    (slot/'model.npz').write_bytes(b'tampered')
    with pytest.raises(Exception):
        d.active_runtime()


def test_signed_runtime_abi_is_enforced(tmp_path):
    r, _, priv, _, kid = signed_registry(tmp_path)
    m = LinearModel([[1,0],[0,1]],[0,0])
    p = tmp_path/'wrong-abi.npz'
    m.save(p)
    man = build_manifest(
        p, '1.0.1', 2, 'float32-linear', 'linear-v999',
        ModelMetrics(1,0,1,100), created_utc='2026-01-01T00:00:00+00:00'
    )
    r.add(p, man, sign_manifest(man, priv), kid)
    d = ABDeployer(tmp_path/'deploy', r, HealthPolicy(min_accuracy=.9,max_ece=1,max_p95_latency_ms=100,max_rss_mb=5000))
    x, y = xy()
    with pytest.raises(HealthGateError, match='runtime ABI mismatch'):
        d.deploy('1.0.1', x, y)


def test_state_path_traversal_is_rejected(tmp_path):
    r, *_ = signed_registry(tmp_path)
    d = ABDeployer(tmp_path/'deploy', r)
    state = d.state().to_dict()
    state['active_slot'] = '../../outside'
    state['slots']['A'] = '1.0.0'
    d.state_path.write_text(json.dumps(state))
    with pytest.raises(Exception):
        d.state()


def test_missing_state_with_existing_slot_fails_closed(tmp_path):
    r, *_ = signed_registry(tmp_path)
    root = tmp_path/'deploy'
    d = ABDeployer(root, r)
    d.state_path.unlink()
    (root/'slots'/'A').mkdir()
    with pytest.raises(Exception):
        ABDeployer(root, r)
