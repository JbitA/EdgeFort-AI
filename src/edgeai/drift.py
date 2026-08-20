from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .errors import ValidationError

@dataclass(frozen=True)
class DriftReport:
    mean_psi: float
    max_psi: float
    fraction_over_threshold: float

class PSIProfile:
    def __init__(self, edges: list[np.ndarray], expected: list[np.ndarray], epsilon=1e-6):
        self.edges=edges;self.expected=expected;self.epsilon=epsilon
    @classmethod
    def fit(cls,x,bins=10):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or len(x)<20 or not np.isfinite(x).all():raise ValidationError('invalid drift reference')
        edges=[];expected=[]
        for j in range(x.shape[1]):
            col=x[:,j];q=np.unique(np.quantile(col,np.linspace(0,1,bins+1)))
            if len(q)<2:q=np.array([col[0]-1e-6,col[0]+1e-6])
            q[0]=-np.inf;q[-1]=np.inf
            counts,_=np.histogram(col,bins=q);p=counts/counts.sum();edges.append(q);expected.append(p)
        return cls(edges,expected)
    def score(self,x,threshold=.2):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or len(x)==0 or x.shape[1]!=len(self.edges) or not np.isfinite(x).all():raise ValidationError('invalid drift batch')
        psis=[]
        for j,(edges,exp) in enumerate(zip(self.edges,self.expected)):
            counts,_=np.histogram(x[:,j],bins=edges);act=counts/max(1,counts.sum());e=np.clip(exp,self.epsilon,None);a=np.clip(act,self.epsilon,None)
            psis.append(float(np.sum((a-e)*np.log(a/e))))
        v=np.array(psis);return DriftReport(float(v.mean()),float(v.max()),float(np.mean(v>threshold)))
