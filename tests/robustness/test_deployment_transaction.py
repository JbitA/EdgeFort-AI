import numpy as np,pytest
from unittest.mock import patch
from tests.helpers import signed_registry
from edgeai.deploy import ABDeployer
from edgeai.health import HealthPolicy

def test_state_write_failure_restores_inactive_slot(tmp_path):
 r,*_=signed_registry(tmp_path,(('1.0.0',1),('1.0.1',2)));x=np.array([[2,0],[0,2]],np.float32);y=np.array([0,1]);d=ABDeployer(tmp_path/'d',r,HealthPolicy(min_accuracy=.9,max_ece=1,max_p95_latency_ms=100,max_rss_mb=5000));d.deploy('1.0.0',x,y)
 orig=d._write
 def fail_on_active(s):
  if s.active_slot=='B':raise OSError('disk fault')
  return orig(s)
 with patch.object(d,'_write',side_effect=fail_on_active):
  with pytest.raises(OSError):d.deploy('1.0.1',x,y)
 assert d.active_version()=='1.0.0'
