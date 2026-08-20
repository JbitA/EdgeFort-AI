from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .errors import ValidationError
from .model_io import (
    DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES,
    model_source,
    preflight_npz,
    validate_linear_headers,
)


class LinearModel:
    ABI = "linear-v1"

    def __init__(self, W, b):
        self.W = np.asarray(W, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        if self.W.ndim != 2 or self.b.ndim != 1 or self.W.shape[0] != self.b.shape[0]:
            raise ValidationError("invalid linear model shape")
        if self.W.shape[0] < 1 or self.W.shape[1] < 1:
            raise ValidationError("linear model dimensions must be non-zero")
        if not (np.isfinite(self.W).all() and np.isfinite(self.b).all()):
            raise ValidationError("non-finite model")

    def logits(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.W.shape[1]:
            raise ValidationError("invalid input shape")
        if not np.isfinite(x).all():
            raise ValidationError("non-finite input")
        return x @ self.W.T + self.b

    def probabilities(self, x):
        z = self.logits(x).astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict(self, x):
        return self.logits(x).argmax(axis=1)

    def save(self, p):
        np.savez_compressed(p, W=self.W, b=self.b, abi=np.array(self.ABI))

    @classmethod
    def load(cls, source, *, max_uncompressed_bytes=DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES):
        with model_source(source) as handle:
            headers = preflight_npz(
                handle,
                {"W", "b", "abi"},
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            validate_linear_headers(headers, quantized=False)
            with np.load(
                handle,
                allow_pickle=False,
                max_header_size=10_000,
            ) as artifact:
                if str(artifact["abi"].item()) != cls.ABI:
                    raise ValidationError("runtime ABI mismatch")
                return cls(artifact["W"], artifact["b"])


def train(x, y, seed=42):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    m = LogisticRegression(max_iter=800, solver="lbfgs", random_state=seed)
    m.fit(x, y)
    return LinearModel(m.coef_, m.intercept_)
