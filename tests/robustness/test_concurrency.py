from concurrent.futures import ThreadPoolExecutor
from tests.integration.test_engine_service import setup

def test_concurrent_inference_state_safe(tmp_path):
 e=setup(tmp_path)
 with ThreadPoolExecutor(max_workers=8) as ex: out=list(ex.map(lambda _:e.predict([[2,0]])[0][0],range(100)))
 assert out==[0]*100 and e.metrics.snapshot()['inference_requests_total']==100
