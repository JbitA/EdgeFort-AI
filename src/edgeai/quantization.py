from __future__ import annotations

import numpy as np

from .errors import ValidationError
from .model_io import (
    DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES,
    model_source,
    preflight_npz,
    validate_linear_headers,
)

# Worst-case signed int8 product is 127*127 for this symmetric quantizer. The
# dynamic runtime explicitly models int32 accumulation, so dimensions above this
# bound could silently wrap in NumPy's int32 matmul and diverge from the checked
# native implementation.
_MAX_SAFE_INT32_DOT = int(np.iinfo(np.int32).max) // (127 * 127)


class WeightOnlyInt8Model:
    ABI = "linear-int8-weight-v1"

    def __init__(self, qW, scales, b):
        self.qW = np.asarray(qW, dtype=np.int8)
        self.scales = np.asarray(scales, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        if (
            self.qW.ndim != 2
            or self.qW.shape[0] < 1
            or self.qW.shape[1] < 1
            or self.scales.shape != (self.qW.shape[0],)
            or self.b.shape != (self.qW.shape[0],)
        ):
            raise ValidationError("invalid int8 model")
        if np.any(self.scales <= 0) or not np.isfinite(self.scales).all():
            raise ValidationError("invalid quantization scale")
        if not np.isfinite(self.b).all():
            raise ValidationError("invalid quantization bias")

    def logits(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.qW.shape[1] or not np.isfinite(x).all():
            raise ValidationError("invalid input")
        return x @ (self.qW.astype(np.float32) * self.scales[:, None]).T + self.b

    def predict(self, x):
        return self.logits(x).argmax(1)

    def save(self, p):
        np.savez_compressed(p, qW=self.qW, scales=self.scales, b=self.b, abi=np.array(self.ABI))

    @classmethod
    def load(cls, source, *, max_uncompressed_bytes=DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES):
        with model_source(source) as handle:
            headers = preflight_npz(
                handle,
                {"qW", "scales", "b", "abi"},
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            validate_linear_headers(headers, quantized=True)
            with np.load(handle, allow_pickle=False, max_header_size=10_000) as artifact:
                if str(artifact["abi"].item()) != cls.ABI:
                    raise ValidationError("runtime ABI mismatch")
                return cls(artifact["qW"], artifact["scales"], artifact["b"])


class DynamicInt8Model(WeightOnlyInt8Model):
    ABI = "linear-int8-dynamic-v1"

    def __init__(self, qW, scales, b):
        super().__init__(qW, scales, b)
        if self.qW.shape[1] > _MAX_SAFE_INT32_DOT:
            raise ValidationError(
                f"dynamic int8 input dimension exceeds safe int32 accumulation bound {_MAX_SAFE_INT32_DOT}"
            )

    def logits(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.qW.shape[1] or not np.isfinite(x).all():
            raise ValidationError("invalid input")
        xscale = np.maximum(np.max(np.abs(x), axis=1) / 127.0, 1e-12).astype(np.float32)
        qx = np.clip(np.rint(x / xscale[:, None]), -127, 127).astype(np.int8)
        accum = qx.astype(np.int32) @ self.qW.astype(np.int32).T
        return accum.astype(np.float32) * (xscale[:, None] * self.scales[None, :]) + self.b


def quantize_weights(model):
    scales = np.maximum(np.max(np.abs(model.W), axis=1) / 127.0, 1e-12).astype(np.float32)
    qweights = np.clip(np.rint(model.W / scales[:, None]), -127, 127).astype(np.int8)
    return WeightOnlyInt8Model(qweights, scales, model.b)


def quantize_dynamic(model):
    scales = np.maximum(np.max(np.abs(model.W), axis=1) / 127.0, 1e-12).astype(np.float32)
    qweights = np.clip(np.rint(model.W / scales[:, None]), -127, 127).astype(np.int8)
    return DynamicInt8Model(qweights, scales, model.b)
