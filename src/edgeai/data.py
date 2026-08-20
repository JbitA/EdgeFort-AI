from __future__ import annotations
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load(seed=42):
    x,y=load_digits(return_X_y=True)
    x=x.astype(np.float32)/16.0
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=seed,stratify=y)
    return x_train,x_test,y_train,y_test

def corrupt(x, sigma=0.10, seed=0):
    rng=np.random.default_rng(seed)
    return np.clip(np.asarray(x,dtype=np.float32)+rng.normal(0,sigma,np.shape(x)),0,1).astype(np.float32)
