import numpy as np,pytest
from edgeai.drift import PSIProfile
from edgeai.errors import ValidationError

def test_drift_low_for_same_distribution():
 r=np.random.default_rng(0);x=r.normal(size=(1000,4));p=PSIProfile.fit(x);d=p.score(r.normal(size=(1000,4)));assert d.mean_psi<.05
def test_drift_detects_shift():
 r=np.random.default_rng(1);x=r.normal(size=(1000,4));p=PSIProfile.fit(x);d=p.score(r.normal(2,size=(1000,4)));assert d.mean_psi>.5 and d.fraction_over_threshold>.5
def test_drift_rejects_bad_input():
 with pytest.raises(ValidationError):PSIProfile.fit([[1,2]])
 r=np.random.default_rng(0);p=PSIProfile.fit(r.normal(size=(100,2)))
 with pytest.raises(ValidationError):p.score([[1,2,3]])

def test_drift_rejects_empty_batch():
 r=np.random.default_rng(0);p=PSIProfile.fit(r.normal(size=(100,2)))
 with pytest.raises(ValidationError):p.score(np.empty((0,2)))
