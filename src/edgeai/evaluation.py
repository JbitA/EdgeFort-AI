from __future__ import annotations
import json,tempfile,time,shutil
from pathlib import Path
import numpy as np
from .data import load,corrupt
from .model import train
from .quantization import quantize_weights,quantize_dynamic
from .metrics import accuracy,expected_calibration_error
from .types import ModelMetrics
from .health import benchmark,HealthPolicy
from .artifact import generate_keypair,public_key_id,build_manifest,sign_manifest,TrustedKeyring
from .registry import Registry
from .deploy import ABDeployer

def _probabilities(logits):
    z=np.asarray(logits,dtype=np.float64); z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

def evaluate(root=None,seed=42):
    xtr,xte,ytr,yte=load(seed); f=train(xtr,ytr,seed); w=quantize_weights(f); d=quantize_dynamic(f)
    if root is None:
        work=Path(tempfile.mkdtemp(prefix='edgeai-eval-'))
    else:
        work=Path(root)
        marker=work/'.edgeai-evaluation-owned'
        if work.exists() and any(work.iterdir()):
            if not marker.exists():
                raise ValueError('refusing to overwrite non-evaluation directory')
            shutil.rmtree(work)
        work.mkdir(parents=True,exist_ok=True)
        marker.write_text('owned by edgeai evaluation\n')
    models=[('float32-linear',f,'float'),('int8-weight-linear',w,'int8_weight'),('int8-dynamic-linear',d,'int8_dynamic')]
    out={}
    priv,pub=generate_keypair(); kid=public_key_id(pub); keyring=TrustedKeyring({kid:pub}); registry=Registry(work/'registry',keyring)
    deployer=ABDeployer(work/'deploy',registry,HealthPolicy(min_accuracy=.93,max_accuracy_drop=.02,max_ece=.25,max_p95_latency_ms=25,max_rss_mb=2048))
    for seq,(kind,m,label) in enumerate(models,1):
        path=work/f'{label}.npz'; m.save(path); bm=benchmark(m,xte,yte,iterations=30)
        probs=_probabilities(m.logits(xte)); clean=accuracy(yte,m.predict(xte)); noisy=accuracy(yte,m.predict(corrupt(xte,.12,seed+100)))
        metrics=ModelMetrics(clean,expected_calibration_error(yte,probs),bm.p95_latency_ms,bm.max_rss_mb)
        man=build_manifest(path,f'1.0.{seq-1}',seq,kind,m.ABI,metrics,{'dataset':'sklearn-digits','seed':seed},created_utc='2026-08-19T00:00:00+00:00')
        sig=sign_manifest(man,priv); registry.add(path,man,sig,kid)
        depmetrics=deployer.deploy(man.version,xte,yte)
        out[label]={'accuracy':clean,'noisy_accuracy_sigma_0_12':noisy,'ece':metrics.ece,'artifact_bytes':path.stat().st_size,'p95_latency_ms_32':bm.p95_latency_ms,'deploy_accuracy':depmetrics.accuracy}
    before=deployer.active_version(); rolled=deployer.rollback(); after=deployer.active_version()
    out['deployment']={'active_before_rollback':before,'rollback_ok':rolled,'active_after_rollback':after,'highest_release_sequence':deployer.state().highest_release_sequence}
    return out

def robustness_campaign(seeds=range(10)):
    rows=[]
    for seed in seeds:
        xtr,xte,ytr,yte=load(42+seed); f=train(xtr,ytr,42+seed); w=quantize_weights(f); d=quantize_dynamic(f)
        row={'seed':seed}
        for name,m in [('float',f),('weight_int8',w),('dynamic_int8',d)]:
            row[name]={'clean':accuracy(yte,m.predict(xte)),'noise_008':accuracy(yte,m.predict(corrupt(xte,.08,900+seed))),'noise_016':accuracy(yte,m.predict(corrupt(xte,.16,1900+seed)))}
        rows.append(row)
    summary={}
    for name in ['float','weight_int8','dynamic_int8']:
        for metric in ['clean','noise_008','noise_016']:
            vals=np.array([r[name][metric] for r in rows],float); summary[f'{name}_{metric}_mean']=float(vals.mean()); summary[f'{name}_{metric}_min']=float(vals.min())
    return {'runs':rows,'summary':summary}
