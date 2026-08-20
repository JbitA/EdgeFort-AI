import numpy as np,pytest
from fastapi.testclient import TestClient
from tests.helpers import signed_registry
from edgeai.deploy import ABDeployer
from edgeai.health import HealthPolicy
from edgeai.engine import InferenceEngine
from edgeai.service import create_app
from edgeai.errors import ValidationError

def setup(tmp_path):
 r,*_=signed_registry(tmp_path);d=ABDeployer(tmp_path/'d',r,HealthPolicy(min_accuracy=.9,max_ece=1,max_p95_latency_ms=100,max_rss_mb=5000));x=np.array([[2,0],[0,2]],np.float32);y=np.array([0,1]);d.deploy('1.0.0',x,y);return InferenceEngine(d,max_batch=4,max_input_features=2)
def test_engine_predict_and_metrics(tmp_path):
 e=setup(tmp_path);p,l,v=e.predict([[2,0]]);assert p.tolist()==[0] and v=='1.0.0' and e.metrics.snapshot()['inference_requests_total']==1
def test_engine_bounds(tmp_path):
 e=setup(tmp_path)
 with pytest.raises(ValidationError):e.predict([[1,2],[1,2],[1,2],[1,2],[1,2]])
 with pytest.raises(ValidationError):e.predict([[1,float('nan')]])
def test_production_app_requires_key(tmp_path):
 e=setup(tmp_path)
 with pytest.raises(RuntimeError):create_app(e)
def test_api_auth_and_prediction(tmp_path):
 c=TestClient(create_app(setup(tmp_path),api_key='secret'))
 assert c.get('/readyz').status_code==401
 assert c.get('/readyz',headers={'x-api-key':'secret'}).status_code==200
 r=c.post('/v1/predict',headers={'x-api-key':'secret'},json={'inputs':[[2,0]]});assert r.status_code==200 and r.json()['predictions']==[0]
def test_api_rejects_extra_and_bad_shape(tmp_path):
 c=TestClient(create_app(setup(tmp_path),api_key='secret'));h={'x-api-key':'secret'}
 assert c.post('/v1/predict',headers=h,json={'inputs':[[2,0]],'x':1}).status_code==422
 assert c.post('/v1/predict',headers=h,json={'inputs':[[1,2,3]]}).status_code==422
def test_development_app(tmp_path):
 c=TestClient(create_app(setup(tmp_path),development=True));assert c.get('/readyz').status_code==200

def test_engine_rejects_ragged_input_as_validation_error(tmp_path):
 e=setup(tmp_path)
 with pytest.raises(ValidationError, match='rectangular'):
  e.predict([[1,2],[3]])


def test_api_ragged_input_is_422_not_500(tmp_path):
 c=TestClient(create_app(setup(tmp_path),api_key='secret'));h={'x-api-key':'secret'}
 r=c.post('/v1/predict',headers=h,json={'inputs':[[1,2],[3]]})
 assert r.status_code==422


def test_production_docs_disabled_development_docs_enabled(tmp_path):
 e=setup(tmp_path)
 prod=TestClient(create_app(e,api_key='secret'))
 assert prod.get('/docs').status_code==404
 dev=TestClient(create_app(e,development=True))
 assert dev.get('/docs').status_code==200

def test_api_rejects_body_before_schema_allocation(tmp_path):
 c=TestClient(create_app(setup(tmp_path),api_key='secret',max_request_bytes=1024));h={'x-api-key':'secret'}
 payload={'inputs':[[float(i) for i in range(200)]]}
 r=c.post('/v1/predict',headers=h,json=payload)
 assert r.status_code==413
