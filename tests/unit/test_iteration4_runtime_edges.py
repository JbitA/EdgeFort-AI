from __future__ import annotations

import asyncio
import io
import os
import pickle
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from edgeai.errors import BackendProcessError, ExecutionDeadlineExceeded, ValidationError
from edgeai.ipc import (
    IPCProtocolError,
    decode_error,
    decode_generation,
    decode_ready,
    decode_result,
    decode_startup_error,
    encode_error,
    encode_generation,
    encode_load_artifact,
    encode_predict,
    encode_result,
    encode_simple,
)
from edgeai.model import LinearModel
from edgeai.runtime import (
    BoundedInferenceExecutor,
    BoundedInferenceQueue,
    _ProcessInferenceRunner,
    _apply_process_memory_limit,
    _isolated_backend_main,
    _socket_recv_frame,
    _socket_send_frame,
    _worker_validate_logits,
)


LIMITS = {"max_output_classes": 4, "max_output_values": 8}
IPC_LIMITS = {
    "max_batch": 4,
    "max_input_features": 4,
    "max_input_values": 16,
    "max_output_classes": 4,
    "max_output_values": 8,
    "max_model_transfer_bytes": 1024 * 1024,
}


class _ArrayFailure:
    def __array__(self, dtype=None, copy=None):
        raise ValueError("no array")


@pytest.mark.parametrize(
    "value,batch,message",
    [
        (_ArrayFailure(), 1, "numeric rank-2"),
        (np.ones(2, np.float32), 1, "rank-2"),
        (np.ones((2, 2), np.float32), 1, "batch dimension"),
        (np.empty((1, 0), np.float32), 1, "class count"),
        (np.ones((1, 5), np.float32), 1, "class count"),
        (np.ones((3, 4), np.float32), 3, "element count"),
        (np.array([["x"]]), 1, "numeric"),
        (np.array([[np.nan]], np.float32), 1, "non-finite"),
    ],
)
def test_worker_output_boundary_rejects_malformed_values(value, batch, message):
    with pytest.raises(ValidationError, match=message):
        _worker_validate_logits(value, batch_size=batch, limits=LIMITS)


def test_worker_output_boundary_normalizes_numeric_values():
    out = _worker_validate_logits(np.array([[1, 2]], np.int16), batch_size=1, limits=LIMITS)
    assert out.dtype == np.float32
    assert out.tolist() == [[1.0, 2.0]]


def _valid_linear_artifact() -> bytes:
    buffer = io.BytesIO()
    LinearModel([[1, 0], [0, 1]], [0, 0]).save(buffer)
    return buffer.getvalue()


def _worker_pair(*, memory_limit_bytes=None):
    parent, child = socket.socketpair()
    thread = threading.Thread(
        target=_isolated_backend_main,
        args=(child, memory_limit_bytes, IPC_LIMITS, 1.0),
        daemon=True,
    )
    thread.start()
    ready = _socket_recv_frame(parent, max_frame_bytes=8192, deadline=time.monotonic() + 1)
    decode_ready(ready)
    return parent, thread


def _send(parent, frame, timeout=1.0):
    _socket_send_frame(parent, frame, deadline=time.monotonic() + timeout)


def _recv(parent, max_bytes=2 * 1024 * 1024, timeout=1.0):
    return _socket_recv_frame(parent, max_frame_bytes=max_bytes, deadline=time.monotonic() + timeout)


def test_isolated_worker_raw_protocol_load_generation_and_result():
    parent, thread = _worker_pair()
    artifact = _valid_linear_artifact()
    try:
        _send(
            parent,
            encode_load_artifact(
                generation=3,
                model_bytes=artifact,
                model_kind="float32-linear",
                runtime_abi=LinearModel.ABI,
                max_uncompressed_bytes=1024 * 1024,
            ),
        )
        assert decode_generation(_recv(parent), "loaded") == 3

        _send(parent, encode_predict(request_id=10, generation=4, inputs=[[1, 0]]))
        response_id, code, message = decode_error(_recv(parent), "error", request=True)
        assert (response_id, code) == (10, "BackendProcessError")
        assert "generation mismatch" in message

        _send(parent, encode_predict(request_id=11, generation=3, inputs=[[2, 0]]))
        logits = decode_result(
            _recv(parent),
            expected_request_id=11,
            expected_batch=1,
            max_output_classes=4,
            max_output_values=8,
        )
        assert logits.shape == (1, 2)
        assert logits[0, 0] > logits[0, 1]
        _send(parent, encode_simple("close"))
    finally:
        parent.close()
        thread.join(1)
    assert not thread.is_alive()


def test_isolated_worker_rejects_corrupt_artifact_without_parent_pickle():
    parent, thread = _worker_pair()
    try:
        _send(
            parent,
            encode_load_artifact(
                generation=2,
                model_bytes=b"not-an-npz",
                model_kind="float32-linear",
                runtime_abi=LinearModel.ABI,
                max_uncompressed_bytes=4096,
            ),
        )
        generation, code, message = decode_error(_recv(parent), "load_error", request=False)
        assert generation == 2
        assert code
        assert "model" in message.lower() or "npz" in message.lower()
        _send(parent, encode_simple("close"))
    finally:
        parent.close()
        thread.join(1)


def test_raw_worker_never_unpickles_peer_messages(tmp_path):
    marker = tmp_path / "pickle-executed"

    class _Exploit:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    malicious_pickle = pickle.dumps(_Exploit())
    parent, thread = _worker_pair()
    try:
        # This is a valid length-prefixed transport message but not an EDGEAI-IPC frame.
        # A recv()/pickle based child would execute the payload while deserializing it.
        _send(parent, malicious_pickle)
        thread.join(1)
        assert not marker.exists()
        assert not thread.is_alive()
    finally:
        parent.close()


def test_process_memory_limit_helper_and_worker_startup_error(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "edgeai.runtime.resource.setrlimit",
        lambda which, value: calls.append((which, value)),
    )
    _apply_process_memory_limit(None)
    _apply_process_memory_limit(512 * 1024 * 1024)
    assert calls and calls[0][1] == (512 * 1024 * 1024, 512 * 1024 * 1024)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="memory limit"):
            _apply_process_memory_limit(value)

    def fail_limit(*_args, **_kwargs):
        raise OSError("rlimit unavailable")

    monkeypatch.setattr("edgeai.runtime._apply_process_memory_limit", fail_limit)
    parent, child = socket.socketpair()
    thread = threading.Thread(
        target=_isolated_backend_main,
        args=(child, 1024, IPC_LIMITS, 1.0),
        daemon=True,
    )
    thread.start()
    try:
        frame = _recv(parent, max_bytes=8192)
        code, message = decode_startup_error(frame)
        assert code == "OSError" and "rlimit unavailable" in message
    finally:
        parent.close()
        thread.join(1)


class _FakeProcess:
    def __init__(self, *, alive=True, survive_terminate=False, start_error=None):
        self.alive = alive
        self.survive_terminate = survive_terminate
        self.start_error = start_error
        self.terminated = 0
        self.killed = 0
        self.closed = 0
        self.started = 0

    def start(self):
        self.started += 1
        if self.start_error:
            raise self.start_error
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated += 1
        if not self.survive_terminate:
            self.alive = False

    def kill(self):
        self.killed += 1
        self.alive = False

    def join(self, timeout=None):
        return None

    def close(self):
        self.closed += 1


def _runner():
    events = []
    return _ProcessInferenceRunner(
        control_timeout_s=0.02,
        memory_limit_bytes=None,
        ipc_limits=IPC_LIMITS,
        counters=events.append,
    ), events


def test_process_runner_abort_escalates_to_kill_and_handles_empty_state():
    runner, _ = _runner()
    process = _FakeProcess(alive=True, survive_terminate=True)
    channel, peer = socket.socketpair()
    runner._process = process
    runner._channel = channel
    try:
        runner.abort()
        assert process.terminated == 1 and process.killed == 1 and process.closed == 1
        assert channel.fileno() == -1
        runner.abort()
    finally:
        peer.close()


def test_process_runner_transport_timeout_eof_and_protocol_paths():
    runner, events = _runner()
    with pytest.raises(BackendProcessError, match="unavailable"):
        runner._receive_frame(deadline=time.monotonic() + .01, deadline_error=False)

    runner, events = _runner()
    channel, peer = socket.socketpair()
    runner._process = _FakeProcess(alive=True)
    runner._channel = channel
    try:
        with pytest.raises(ExecutionDeadlineExceeded):
            runner._receive_frame(deadline=time.monotonic() + .001, deadline_error=True)
        assert "execution_timeouts" in events
    finally:
        peer.close()

    runner, events = _runner()
    channel, peer = socket.socketpair()
    runner._process = _FakeProcess(alive=True)
    runner._channel = channel
    peer.close()
    with pytest.raises(BackendProcessError, match="exited or violated IPC protocol"):
        runner._receive_frame(deadline=time.monotonic() + .1, deadline_error=False)
    assert "process_crashes" in events

    runner, events = _runner()
    channel, peer = socket.socketpair()
    runner._process = _FakeProcess(alive=True)
    runner._channel = channel
    try:
        # Oversized declared frame is rejected before allocating its body.
        peer.sendall((10_000_000).to_bytes(8, "big"))
        with pytest.raises(BackendProcessError, match="violated IPC protocol"):
            runner._receive_frame(deadline=time.monotonic() + .1, deadline_error=False)
        assert "process_crashes" in events
    finally:
        peer.close()


def test_process_runner_execute_response_validation(monkeypatch):
    prepared = SimpleNamespace(
        generation=7,
        inputs=np.array([[1, 0]], np.float32),
        model_bytes=_valid_linear_artifact(),
        model_kind="float32-linear",
        runtime_abi=LinearModel.ABI,
        max_uncompressed_bytes=1024 * 1024,
    )

    def configured(reply, *, generation=7):
        runner, events = _runner()
        runner._generation = generation
        monkeypatch.setattr(runner, "start", lambda: None)
        monkeypatch.setattr(runner, "_send_frame", lambda *args, **kwargs: None)
        monkeypatch.setattr(runner, "_receive_frame", lambda **kwargs: reply)
        monkeypatch.setattr(runner, "abort", lambda: None)
        return runner, events

    runner, events = configured(b"not-a-frame")
    with pytest.raises(BackendProcessError, match="invalid response"):
        runner.execute(prepared, timeout_s=1, limits=IPC_LIMITS)
    assert "process_crashes" in events

    runner, _ = configured(
        encode_error("error", request_id=1, generation=None, code="ValidationError", message="bad")
    )
    with pytest.raises(ValidationError, match="bad"):
        runner.execute(prepared, timeout_s=1, limits=IPC_LIMITS)

    runner, _ = configured(
        encode_error("error", request_id=1, generation=None, code="RuntimeError", message="bad")
    )
    with pytest.raises(BackendProcessError, match="RuntimeError"):
        runner.execute(prepared, timeout_s=1, limits=IPC_LIMITS)

    runner, _ = configured(
        encode_result(request_id=1, logits=np.array([[1, 2]], np.float32))
    )
    result = runner.execute(prepared, timeout_s=1, limits=IPC_LIMITS)
    assert result.tolist() == [[1.0, 2.0]]

    runner, _ = configured(
        encode_error("load_error", request_id=None, generation=7, code="HealthGateError", message="bad artifact"),
        generation=None,
    )
    with pytest.raises(BackendProcessError, match="bad artifact"):
        runner.execute(prepared, timeout_s=1, limits=IPC_LIMITS)

    runner, events = configured(encode_generation("loaded", 8), generation=None)
    with pytest.raises(BackendProcessError, match="rejected runtime"):
        runner.execute(prepared, timeout_s=1, limits=IPC_LIMITS)
    assert "process_crashes" in events


def test_process_runner_close_and_alive_paths():
    runner, _ = _runner()
    assert not runner.is_alive()
    process = _FakeProcess(alive=True)
    channel, peer = socket.socketpair()
    runner._process = process
    runner._channel = channel
    assert runner.is_alive()
    runner.close()
    assert not runner.is_alive()
    peer.close()


def test_queue_additional_edge_paths():
    queue = BoundedInferenceQueue(2)
    with pytest.raises(ValueError):
        queue.pop_with_expired(-1)
    payload = object()
    assert not queue.expire_payload(payload)
    queue.push(payload)
    assert queue.expire_payload(payload)
    assert queue.drain() == []
    queue.wake_all()


class _ImmediateEngine:
    def predict(self, payload):
        return payload


def test_executor_additional_validation_and_closed_paths():
    for workers in (0, 65, True):
        with pytest.raises(ValueError, match="workers"):
            BoundedInferenceExecutor(_ImmediateEngine(), workers=workers)
    for value in (0, -1, float("inf"), True):
        with pytest.raises(ValueError, match="execution_timeout"):
            BoundedInferenceExecutor(_ImmediateEngine(), execution_mode="process", execution_timeout_s=value)
    with pytest.raises(ValueError, match="authenticated process isolation"):
        BoundedInferenceExecutor(_ImmediateEngine(), execution_mode="process", execution_timeout_s=1)

    executor = BoundedInferenceExecutor(_ImmediateEngine())
    assert executor.ensure_ready()["execution_mode"] == "thread"
    assert executor.close(wait=True, timeout_s=1)
    with pytest.raises(Exception):
        executor.start()
    for value in (-1, float("inf"), True):
        with pytest.raises(ValueError, match="timeout_s"):
            executor.close(timeout_s=value)


def test_async_submit_cancellation_removes_queued_request():
    class BlockingEngine:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def predict(self, payload):
            if payload == "block":
                self.started.set()
                self.release.wait(2)
            return payload

    engine = BlockingEngine()
    executor = BoundedInferenceExecutor(engine, capacity=2, workers=1)

    async def scenario():
        first = asyncio.create_task(executor.submit_async("block"))
        while not engine.started.is_set():
            await asyncio.sleep(.005)
        queued = asyncio.create_task(executor.submit_async("cancel"))
        while executor.stats()["depth"] < 1:
            await asyncio.sleep(.005)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert executor.stats()["depth"] == 0
        engine.release.set()
        assert await first == "block"

    asyncio.run(scenario())
    executor.close()
