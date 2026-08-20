from __future__ import annotations
import numpy as np

def accuracy(y,p): return float(np.mean(np.asarray(y)==np.asarray(p)))

def expected_calibration_error(y, probs, bins=10):
    y=np.asarray(y); probs=np.asarray(probs,float)
    conf=probs.max(1); pred=probs.argmax(1); correct=(pred==y).astype(float); ece=0.0
    edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-12))
        if mask.any(): ece += float(mask.mean())*abs(float(correct[mask].mean())-float(conf[mask].mean()))
    return float(ece)
