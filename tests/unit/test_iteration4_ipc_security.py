from __future__ import annotations

import json

import numpy as np
import pytest

from edgeai.ipc import (
    IPC_MAGIC,
    IPC_VERSION,
    MAX_HEADER_BYTES,
    IPCProtocolError,
    decode_error,
    decode_frame,
    decode_generation,
    decode_load_artifact,
    decode_predict,
    decode_ready,
    decode_result,
    decode_simple,
    decode_startup_error,
    encode_error,
    encode_frame,
    encode_generation,
    encode_load_artifact,
    encode_predict,
    encode_ready,
    encode_result,
    encode_simple,
    encode_startup_error,
    frame_length_prefix,
    parse_frame_length,
    require_fields,
)


def _raw(header: dict, payload: bytes = b"") -> bytes:
    return json.dumps(header, separators=(",", ":"), allow_nan=True).encode() + b"\n" + payload


def _base(**updates):
    value = {"magic": IPC_MAGIC, "op": "x", "payload_bytes": 0, "v": IPC_VERSION}
    value.update(updates)
    return value


@pytest.mark.parametrize("op", [None, "", "x" * 33, "π"])
def test_encode_frame_rejects_invalid_operations(op):
    with pytest.raises(IPCProtocolError, match="operation"):
        encode_frame(op)  # type: ignore[arg-type]


def test_encode_frame_rejects_nonbytes_reserved_bad_keys_and_noncanonical_values():
    with pytest.raises(IPCProtocolError, match="payload"):
        encode_frame("x", payload=bytearray(b"x"))  # type: ignore[arg-type]
    with pytest.raises(IPCProtocolError, match="reserved"):
        encode_frame("x", {"v": 9})
    with pytest.raises(IPCProtocolError, match="keys"):
        encode_frame("x", {1: "bad"})  # type: ignore[dict-item]
    with pytest.raises(IPCProtocolError, match="keys"):
        encode_frame("x", {"": "bad"})
    with pytest.raises(IPCProtocolError, match="canonical JSON"):
        encode_frame("x", {"bad": float("nan")})
    with pytest.raises(IPCProtocolError, match="header exceeds"):
        encode_frame("x", {"huge": "a" * (MAX_HEADER_BYTES + 1)})


@pytest.mark.parametrize("limit", [True, -1, 1.5, "4"])
def test_decode_frame_validates_limit_type(limit):
    with pytest.raises(ValueError, match="max_payload_bytes"):
        decode_frame(encode_simple("x"), max_payload_bytes=limit)  # type: ignore[arg-type]


def test_decode_frame_rejects_malformed_headers_and_payloads():
    cases = [
        (bytearray(b"x"), "frame must be bytes"),
        (b"no-newline", "terminator"),
        (b"\n", "empty header"),
        (b"\xff\n", "JSON header"),
        (b"{bad}\n", "JSON header"),
        (b"[]\n", "header must be an object"),
        (_raw({"magic": IPC_MAGIC}), "missing IPC header field"),
        (_raw(_base(magic="wrong")), "unsupported IPC protocol"),
        (_raw(_base(v=99)), "unsupported IPC protocol"),
        (_raw(_base(op="")), "text field op"),
        (_raw(_base(payload_bytes=True)), "integer field payload_bytes"),
        (_raw(_base(payload_bytes=2), b"x"), "payload length mismatch"),
    ]
    for frame, message in cases:
        with pytest.raises(IPCProtocolError, match=message):
            decode_frame(frame, max_payload_bytes=8)  # type: ignore[arg-type]


def test_decode_frame_rejects_duplicate_keys_nonfinite_and_payload_over_bound():
    duplicate = (
        b'{"magic":"EDGEAI-IPC","magic":"EDGEAI-IPC","op":"x",'
        b'"payload_bytes":0,"v":1}\n'
    )
    with pytest.raises(IPCProtocolError, match="duplicate"):
        decode_frame(duplicate, max_payload_bytes=0)
    nonfinite = b'{"magic":"EDGEAI-IPC","op":"x","payload_bytes":NaN,"v":1}\n'
    with pytest.raises(IPCProtocolError, match="non-finite"):
        decode_frame(nonfinite, max_payload_bytes=0)
    with pytest.raises(IPCProtocolError, match="integer field payload_bytes"):
        decode_frame(_raw(_base(payload_bytes=2), b"xx"), max_payload_bytes=1)


def test_require_fields_is_exact():
    header, _ = decode_frame(encode_frame("x", {"a": 1}), max_payload_bytes=0)
    require_fields(header, "a")
    with pytest.raises(IPCProtocolError, match="fields mismatch"):
        require_fields(header)


def test_ready_simple_generation_and_error_frames_are_strict():
    assert decode_ready(encode_ready(123)) == 123
    wrong_ready = encode_frame("close", {"pid": 123})
    with pytest.raises(IPCProtocolError, match="expected ready"):
        decode_ready(wrong_ready)
    bad_pid = encode_frame("ready", {"pid": 0})
    with pytest.raises(IPCProtocolError, match="worker pid"):
        decode_ready(bad_pid)

    decode_simple(encode_simple("close"), "close")
    with pytest.raises(IPCProtocolError, match="expected close"):
        decode_simple(encode_simple("other"), "close")

    assert decode_generation(encode_generation("loaded", 7), "loaded") == 7
    with pytest.raises(IPCProtocolError, match="expected loaded"):
        decode_generation(encode_generation("other", 7), "loaded")

    frame = encode_error("error", request_id=5, generation=None, code="X", message="")
    assert decode_error(frame, "error", request=True) == (5, "X", "backend failure")
    with pytest.raises(IPCProtocolError, match="expected other"):
        decode_error(frame, "other", request=True)
    load = encode_error("load_error", request_id=None, generation=8, code="Bad", message="no")
    assert decode_error(load, "load_error", request=False) == (8, "Bad", "no")

    startup = encode_startup_error("OSError", "no")
    assert decode_startup_error(startup) == ("OSError", "no")
    wrong_startup = encode_frame("error", {"code": "X", "message": "x"})
    with pytest.raises(IPCProtocolError, match="expected startup_error"):
        decode_startup_error(wrong_startup)


def test_text_field_bounds_and_utf8_rejection():
    with pytest.raises(IPCProtocolError, match="text field code"):
        encode_error("error", request_id=1, generation=None, code="x" * 65, message="x")
    with pytest.raises(IPCProtocolError, match="text field message"):
        encode_error("error", request_id=1, generation=None, code="x", message="x" * 4097)
    # Lone surrogate cannot be encoded as UTF-8.
    with pytest.raises(IPCProtocolError, match="UTF-8"):
        encode_error("error", request_id=1, generation=None, code="x", message="\ud800")


def test_load_artifact_frame_bounds_and_schema():
    model = b"1234"
    frame = encode_load_artifact(
        generation=2,
        model_bytes=model,
        model_kind="float32-linear",
        runtime_abi="edgeai.linear.float32.v1",
        max_uncompressed_bytes=4096,
    )
    got = decode_load_artifact(frame, max_model_bytes=8192)
    assert got[0] == 2 and got[1] == model and got[-1] == 4096

    for value in (b"", bytearray(b"x")):
        with pytest.raises(IPCProtocolError, match="artifact payload"):
            encode_load_artifact(
                generation=1,
                model_bytes=value,  # type: ignore[arg-type]
                model_kind="x",
                runtime_abi="y",
                max_uncompressed_bytes=1024,
            )
    with pytest.raises(IPCProtocolError, match="fields mismatch|expected load_artifact"):
        decode_load_artifact(encode_simple("close"), max_model_bytes=8192)
    tiny_bound = encode_load_artifact(
        generation=1,
        model_bytes=b"x",
        model_kind="x",
        runtime_abi="y",
        max_uncompressed_bytes=1,
    )
    with pytest.raises(IPCProtocolError, match="uncompressed bound"):
        decode_load_artifact(tiny_bound, max_model_bytes=8192)
    too_large_bound = encode_load_artifact(
        generation=1,
        model_bytes=b"x",
        model_kind="x",
        runtime_abi="y",
        max_uncompressed_bytes=8193,
    )
    with pytest.raises(IPCProtocolError, match="uncompressed bound"):
        decode_load_artifact(too_large_bound, max_model_bytes=8192)


class _BadArray:
    def __array__(self, dtype=None, copy=None):
        raise ValueError("bad")


def test_predict_tensor_encoding_and_decoding_is_strict():
    with pytest.raises(IPCProtocolError, match="float32"):
        encode_predict(request_id=1, generation=1, inputs=_BadArray())
    with pytest.raises(IPCProtocolError, match="rank-2"):
        encode_predict(request_id=1, generation=1, inputs=[1, 2])

    frame = encode_predict(request_id=3, generation=4, inputs=[[1, 2], [3, 4]])
    request_id, generation, array = decode_predict(
        frame, max_batch=2, max_input_features=2, max_input_values=4
    )
    assert (request_id, generation) == (3, 4)
    assert array.dtype == np.float32 and array.tolist() == [[1, 2], [3, 4]]

    header, payload = decode_frame(frame, max_payload_bytes=16)
    variants = [
        (encode_frame("other", {k: v for k, v in header.items() if k not in {"magic", "op", "payload_bytes", "v"}}, payload), "expected predict"),
        (encode_frame("predict", {"cols": 2, "generation": 4, "request_id": 3, "rows": 0}, b""), "outside configured bounds"),
        (encode_frame("predict", {"cols": 3, "generation": 4, "request_id": 3, "rows": 1}, np.zeros(3, dtype="<f4").tobytes()), "integer field cols"),
        (encode_frame("predict", {"cols": 2, "generation": 4, "request_id": 3, "rows": 2}, b"1234"), "byte length mismatch"),
        (encode_frame("predict", {"cols": 2, "generation": 4, "request_id": 3, "rows": 2}, np.array([1, 2, 3, np.nan], dtype="<f4").tobytes()), "non-finite"),
    ]
    for bad, message in variants:
        with pytest.raises(IPCProtocolError, match=message):
            decode_predict(bad, max_batch=2, max_input_features=2, max_input_values=4)

    over_product = encode_frame(
        "predict",
        {"cols": 2, "generation": 4, "request_id": 3, "rows": 2},
        b"",
    )
    with pytest.raises(IPCProtocolError, match="outside configured bounds"):
        decode_predict(over_product, max_batch=3, max_input_features=3, max_input_values=3)


def test_result_tensor_encoding_and_decoding_is_strict():
    with pytest.raises(IPCProtocolError, match="rank-2"):
        encode_result(request_id=1, logits=[1, 2])
    frame = encode_result(request_id=9, logits=[[1, 2]])
    result = decode_result(
        frame,
        expected_request_id=9,
        expected_batch=1,
        max_output_classes=2,
        max_output_values=2,
    )
    assert result.tolist() == [[1, 2]]

    with pytest.raises(IPCProtocolError, match="request ID mismatch"):
        decode_result(frame, expected_request_id=8, expected_batch=1, max_output_classes=2, max_output_values=2)

    cases = [
        (encode_frame("other", {"cols": 2, "request_id": 9, "rows": 1}, np.zeros(2, dtype="<f4").tobytes()), "expected result"),
        (encode_frame("result", {"cols": 2, "request_id": 9, "rows": 0}, b""), "outside configured bounds"),
        (encode_frame("result", {"cols": 0, "request_id": 9, "rows": 1}, b""), "outside configured bounds"),
        (encode_frame("result", {"cols": 2, "request_id": 9, "rows": 1}, b"1234"), "byte length mismatch"),
        (encode_frame("result", {"cols": 2, "request_id": 9, "rows": 1}, np.array([1, np.inf], dtype="<f4").tobytes()), "non-finite"),
    ]
    for bad, message in cases:
        with pytest.raises(IPCProtocolError, match=message):
            decode_result(bad, expected_request_id=9, expected_batch=1, max_output_classes=2, max_output_values=2)

    product = encode_frame(
        "result",
        {"cols": 2, "request_id": 9, "rows": 2},
        b"",
    )
    with pytest.raises(IPCProtocolError, match="outside configured bounds"):
        decode_result(product, expected_request_id=9, expected_batch=2, max_output_classes=2, max_output_values=3)


def test_frame_length_prefix_validation():
    prefix = frame_length_prefix(123)
    assert parse_frame_length(prefix) == 123
    for value in (-1, True, 1.5, 1 << 63):
        with pytest.raises(IPCProtocolError, match="frame size"):
            frame_length_prefix(value)  # type: ignore[arg-type]
    with pytest.raises(IPCProtocolError, match="prefix"):
        parse_frame_length(b"short")
