from __future__ import annotations
import threading,time
from collections import defaultdict
class Metrics:
    def __init__(self): self._lock=threading.Lock(); self._c=defaultdict(float); self._lat=[]
    def inc(self,name,value=1.0):
        with self._lock:self._c[name]+=value
    def observe_latency_ms(self,v):
        with self._lock:self._lat.append(float(v)); self._lat=self._lat[-10000:]
    def snapshot(self):
        with self._lock:
            lat=sorted(self._lat); p95=lat[int(.95*(len(lat)-1))] if lat else 0.0
            return {**dict(self._c),'inference_p95_ms':p95,'inference_samples':len(lat)}
