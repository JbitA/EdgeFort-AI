from edgeai.data import load
from edgeai.model import train
from edgeai.quantization import quantize_dynamic
from edgeai.health import benchmark

def test_dynamic_int8_p95_budget():
 xtr,xte,ytr,yte=load();m=quantize_dynamic(train(xtr,ytr));r=benchmark(m,xte,yte,iterations=60,batch=32);assert r.p95_latency_ms<5.0
