from __future__ import annotations

import socket
import threading
import time

import numpy as np

from edgeai.health import HealthPolicy
from edgeai.ipc import (
    MAX_HEADER_BYTES,
    decode_error,
    decode_generation,
    decode_metrics,
    decode_ready,
    encode_load_artifact,
    encode_qualification,
    encode_simple,
)
from edgeai.model import LinearModel
from edgeai.qualification import QualificationLimits, _qualification_worker
from edgeai.runtime import _socket_recv_frame, _socket_send_frame


def _model_bytes(tmp_path):
    path = tmp_path / "model.npz"
    LinearModel([[1, 0], [0, 1]], [0, 0]).save(path)
    return path.read_bytes()


def _start_worker(monkeypatch, *, max_model_bytes=1024 * 1024, sandbox_error=None):
    if sandbox_error is None:
        monkeypatch.setattr("edgeai.qualification.apply_process_sandbox", lambda **kwargs: None)
    else:
        def fail(**kwargs):
            raise sandbox_error
        monkeypatch.setattr("edgeai.qualification.apply_process_sandbox", fail)
    parent, child = socket.socketpair()
    limits = QualificationLimits(
        timeout_s=2,
        max_validation_rows=16,
        max_validation_features=16,
        max_validation_values=64,
        iterations=2,
        batch=1,
        nofile_limit=64,
    )
    thread = threading.Thread(
        target=_qualification_worker,
        kwargs={"channel": child, "max_model_bytes": max_model_bytes, "limits": limits},
        daemon=True,
    )
    thread.start()
    return parent, thread, limits


def _recv(sock):
    return _socket_recv_frame(
        sock,
        max_frame_bytes=MAX_HEADER_BYTES + 1 + 64 * 1024,
        deadline=time.monotonic() + 2,
    )


def _send(sock, frame):
    _socket_send_frame(sock, frame, deadline=time.monotonic() + 2)


def test_qualification_worker_protocol_happy_path_in_parent_tracer(monkeypatch, tmp_path):
    parent, thread, limits = _start_worker(monkeypatch)
    try:
        assert decode_ready(_recv(parent)) > 0
        raw = _model_bytes(tmp_path)
        _send(
            parent,
            encode_load_artifact(
                generation=9,
                model_bytes=raw,
                model_kind="float32-linear",
                runtime_abi=LinearModel.ABI,
                max_uncompressed_bytes=1024 * 1024,
            ),
        )
        assert decode_generation(_recv(parent), "loaded") == 9
        _send(
            parent,
            encode_qualification(
                inputs=np.array([[2, 0], [0, 2]], np.float32),
                labels=np.array([0, 1], np.int64),
                iterations=limits.iterations,
                batch=limits.batch,
                max_validation_rows=limits.max_validation_rows,
                max_validation_features=limits.max_validation_features,
                max_validation_values=limits.max_validation_values,
            ),
        )
        metrics = decode_metrics(_recv(parent))
        assert metrics.accuracy == 1.0
    finally:
        parent.close()
        thread.join(2)
        assert not thread.is_alive()


def test_qualification_worker_reports_sandbox_setup_failure(monkeypatch):
    parent, thread, _ = _start_worker(monkeypatch, sandbox_error=OSError("sandbox denied"))
    try:
        identifier, code, message = decode_error(_recv(parent), "qualification_error", request=False)
        assert identifier == 0
        assert code == "OSError"
        assert "sandbox denied" in message
    finally:
        parent.close()
        thread.join(2)
        assert not thread.is_alive()


def test_qualification_worker_reports_model_load_failure(monkeypatch):
    parent, thread, _ = _start_worker(monkeypatch)
    try:
        decode_ready(_recv(parent))
        _send(
            parent,
            encode_load_artifact(
                generation=1,
                model_bytes=b"not-a-valid-model",
                model_kind="float32-linear",
                runtime_abi=LinearModel.ABI,
                max_uncompressed_bytes=1024 * 1024,
            ),
        )
        identifier, code, message = decode_error(_recv(parent), "qualification_error", request=False)
        assert identifier == 0
        assert code
        assert message
    finally:
        parent.close()
        thread.join(2)
        assert not thread.is_alive()


def test_qualification_worker_reports_invalid_benchmark_request(monkeypatch, tmp_path):
    parent, thread, _ = _start_worker(monkeypatch)
    try:
        decode_ready(_recv(parent))
        raw = _model_bytes(tmp_path)
        _send(
            parent,
            encode_load_artifact(
                generation=2,
                model_bytes=raw,
                model_kind="float32-linear",
                runtime_abi=LinearModel.ABI,
                max_uncompressed_bytes=1024 * 1024,
            ),
        )
        assert decode_generation(_recv(parent), "loaded") == 2
        _send(parent, encode_simple("not-qualification"))
        identifier, code, message = decode_error(_recv(parent), "qualification_error", request=False)
        assert identifier == 0
        assert code
        assert message
    finally:
        parent.close()
        thread.join(2)
        assert not thread.is_alive()


def test_qualification_worker_treats_broken_parent_channel_as_fatal(monkeypatch):
    parent, thread, _ = _start_worker(monkeypatch)
    decode_ready(_recv(parent))
    parent.close()
    thread.join(2)
    assert not thread.is_alive()
