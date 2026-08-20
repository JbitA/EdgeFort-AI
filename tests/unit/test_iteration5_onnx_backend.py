from __future__ import annotations

import io
import sys
import types

import numpy as np
import pytest

from edgeai.errors import HealthGateError, ValidationError
from edgeai.health import expected_runtime_abi, load_runtime
from edgeai.onnx_backend import OnnxRuntimeCPUModel


class _Node:
    def __init__(self, name, type_, shape):
        self.name = name
        self.type = type_
        self.shape = shape


class _Options:
    def __init__(self):
        self.entries = {}
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.execution_mode = None
        self.graph_optimization_level = None
        self.enable_cpu_mem_arena = None
        self.enable_mem_pattern = None

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class _Session:
    input_meta = _Node("input", "tensor(float)", [None, 2])
    output_meta = _Node("logits", "tensor(float)", [None, 2])
    providers = ["CPUExecutionProvider"]
    last_options = None
    last_model = None

    def __init__(self, model, *, sess_options, providers):
        type(self).last_model = model
        type(self).last_options = sess_options
        self.requested_providers = providers

    def get_inputs(self):
        return [self.input_meta]

    def get_outputs(self):
        return [self.output_meta]

    def get_providers(self):
        return list(self.providers)

    def run(self, outputs, feed):
        assert outputs == ["logits"]
        x = feed["input"]
        return [np.column_stack((x[:, 0], -x[:, 0])).astype(np.float32)]


def _fake_ort(monkeypatch, session_type=_Session):
    fake = types.SimpleNamespace(
        SessionOptions=_Options,
        InferenceSession=session_type,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_BASIC="basic"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return fake


def test_onnx_runtime_cpu_backend_is_narrow_and_deterministic(monkeypatch):
    _fake_ort(monkeypatch)
    runtime = load_runtime(
        io.BytesIO(b"fake-onnx"),
        "onnxruntime-cpu",
        OnnxRuntimeCPUModel.ABI,
        max_uncompressed_bytes=4096,
    )
    assert expected_runtime_abi("onnxruntime-cpu") == "onnxruntime-cpu-v1"
    logits = runtime.logits([[2, 0], [-1, 0]])
    assert logits.tolist() == [[2.0, -2.0], [-1.0, 1.0]]
    assert runtime.predict([[2, 0], [-1, 0]]).tolist() == [0, 1]
    assert _Session.last_model == b"fake-onnx"
    options = _Session.last_options
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == "sequential"
    assert options.graph_optimization_level == "basic"
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False
    assert options.entries == {
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0",
    }


def test_onnx_backend_rejects_provider_fallback(monkeypatch):
    class Fallback(_Session):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    _fake_ort(monkeypatch, Fallback)
    with pytest.raises(ValidationError, match="exclusively"):
        OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)


@pytest.mark.parametrize(
    "input_meta,output_meta,message",
    [
        (_Node("x", "tensor(double)", [None, 2]), _Node("y", "tensor(float)", [None, 2]), "input must"),
        (_Node("x", "tensor(float)", [None, 2]), _Node("y", "tensor(int64)", [None, 2]), "output must"),
        (_Node("x", "tensor(float)", [None, None]), _Node("y", "tensor(float)", [None, 2]), "feature"),
        (_Node("x", "tensor(float)", [None, 2]), _Node("y", "tensor(float)", [None, None]), "class"),
        (_Node("x", "tensor(float)", [2, 2]), _Node("y", "tensor(float)", [3, 2]), "disagree"),
        (_Node("x", "tensor(float)", [None, 2, 1]), _Node("y", "tensor(float)", [None, 2]), "rank-2"),
    ],
)
def test_onnx_backend_rejects_ambiguous_tensor_contract(monkeypatch, input_meta, output_meta, message):
    class Bad(_Session):
        pass

    Bad.input_meta = input_meta
    Bad.output_meta = output_meta
    _fake_ort(monkeypatch, Bad)
    with pytest.raises(ValidationError, match=message):
        OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)


def test_onnx_backend_enforces_fixed_batch_and_runtime_output_contract(monkeypatch):
    class Fixed(_Session):
        input_meta = _Node("input", "tensor(float)", [1, 2])
        output_meta = _Node("logits", "tensor(float)", [1, 2])

    _fake_ort(monkeypatch, Fixed)
    runtime = OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)
    with pytest.raises(ValidationError, match="fixed ONNX batch"):
        runtime.logits([[1, 0], [2, 0]])
    with pytest.raises(ValidationError, match="non-finite"):
        runtime.logits([[np.inf, 0]])

    class BadRun(Fixed):
        def run(self, outputs, feed):
            return []

    _fake_ort(monkeypatch, BadRun)
    runtime = OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)
    with pytest.raises(ValidationError, match="output collection"):
        runtime.logits([[1, 0]])


def test_onnx_backend_fails_cleanly_without_optional_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    with pytest.raises(HealthGateError, match="optional 'onnx' dependency"):
        OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)


def test_onnx_backend_rejects_empty_or_oversized_serialized_model(monkeypatch):
    _fake_ort(monkeypatch)
    with pytest.raises(ValidationError, match="empty"):
        OnnxRuntimeCPUModel.load(io.BytesIO(b""), max_uncompressed_bytes=1024)
    with pytest.raises(ValidationError, match="exceeds"):
        OnnxRuntimeCPUModel.load(io.BytesIO(b"x" * 1025), max_uncompressed_bytes=1024)



def test_onnx_backend_additional_fail_closed_edges(monkeypatch):
    _fake_ort(monkeypatch)
    for invalid in (True, 100, 1.5):
        with pytest.raises(ValueError, match="max_uncompressed_bytes"):
            OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=invalid)

    class BadBatch(_Session):
        input_meta = _Node("input", "tensor(float)", [0, 2])
    _fake_ort(monkeypatch, BadBatch)
    with pytest.raises(ValidationError, match="batch dimension"):
        OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)

    class BadInputName(_Session):
        input_meta = _Node("", "tensor(float)", [None, 2])
    _fake_ort(monkeypatch, BadInputName)
    with pytest.raises(ValidationError, match="input name"):
        OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)

    class BadOutputName(_Session):
        output_meta = _Node("", "tensor(float)", [None, 2])
    _fake_ort(monkeypatch, BadOutputName)
    with pytest.raises(ValidationError, match="output name"):
        OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)

    class ConstructorFailure(_Session):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ORT parser failure")
    _fake_ort(monkeypatch, ConstructorFailure)
    with pytest.raises(ValidationError, match="session rejected model"):
        OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)


def test_onnx_backend_runtime_failure_and_shape_edges(monkeypatch):
    class FixedOutput(_Session):
        input_meta = _Node("input", "tensor(float)", [None, 2])
        output_meta = _Node("logits", "tensor(float)", [1, 2])
    _fake_ort(monkeypatch, FixedOutput)
    runtime = OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)
    with pytest.raises(ValidationError, match="fixed ONNX output batch"):
        runtime.logits([[1, 0], [2, 0]])
    with pytest.raises(ValidationError, match="input shape"):
        runtime.logits([[1, 0, 3]])
    with pytest.raises(ValidationError, match="invalid ONNX input tensor"):
        runtime.logits([[object(), 0]])

    class RunFailure(_Session):
        def run(self, outputs, feed):
            raise RuntimeError("native execution failed")
    _fake_ort(monkeypatch, RunFailure)
    runtime = OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)
    with pytest.raises(ValidationError, match="inference failed"):
        runtime.logits([[1, 0]])

    class WrongShape(_Session):
        def run(self, outputs, feed):
            return [np.array([1.0, 2.0], np.float32)]
    _fake_ort(monkeypatch, WrongShape)
    runtime = OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)
    with pytest.raises(ValidationError, match="invalid logits shape"):
        runtime.predict([[1, 0]])


def test_onnx_backend_requires_exact_single_io(monkeypatch):
    class TooManyInputs(_Session):
        def get_inputs(self):
            return [self.input_meta, self.input_meta]
    _fake_ort(monkeypatch, TooManyInputs)
    with pytest.raises(ValidationError, match="exactly one input"):
        OnnxRuntimeCPUModel.load(io.BytesIO(b"model"), max_uncompressed_bytes=4096)
