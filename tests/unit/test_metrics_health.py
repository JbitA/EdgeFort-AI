import numpy as np,pytest
from edgeai.metrics import expected_calibration_error,accuracy
from edgeai.types import ModelMetrics
from edgeai.health import enforce,HealthPolicy
from edgeai.errors import HealthGateError

def test_ece_perfect_confidence():
    y=np.array([0,1]);p=np.array([[1,0],[0,1]],float);assert expected_calibration_error(y,p)<1e-12
def test_accuracy(): assert accuracy([1,2],[1,0])==.5
@pytest.mark.parametrize('m', [ModelMetrics(.8,.1,1,1),ModelMetrics(.95,.9,1,1),ModelMetrics(.95,.1,99,1),ModelMetrics(.95,.1,1,9999)])
def test_health_rejects_bad_metrics(m):
    with pytest.raises(HealthGateError): enforce(HealthPolicy(min_accuracy=.9,max_ece=.2,max_p95_latency_ms=10,max_rss_mb=100),m)
def test_health_rejects_regression():
    with pytest.raises(HealthGateError):enforce(HealthPolicy(min_accuracy=.8,max_accuracy_drop=.01),ModelMetrics(.9,.1,1,1),ModelMetrics(.95,.1,1,1))
@pytest.mark.parametrize('kwargs',[{'min_accuracy':1.1},{'max_accuracy_drop':-1},{'max_ece':2},{'max_p95_latency_ms':0},{'max_rss_mb':0}])
def test_health_policy_rejects_invalid(kwargs):
    with pytest.raises(ValueError): HealthPolicy(**kwargs)
