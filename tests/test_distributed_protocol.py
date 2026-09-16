import os
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from kernelleaf.distributed import (
    CodecError,
    FramedTransport,
    InvalidFrameLength,
    JsonCodec,
    Message,
    MessageCodec,
    MessageType,
    PeerDisconnected,
    ProtocolError,
    TransportTimeout,
    pack_frame,
    receive_frame,
)


def _message(index=1, payload=None):
    return Message(
        message_type=MessageType.PUSH_GRADIENTS,
        request_id=f"request-{index}",
        worker_id="worker-0",
        step=index,
        payload={} if payload is None else payload,
    )


def _tcp_pair():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname(), timeout=1)
    server, _ = listener.accept()
    listener.close()
    client.settimeout(None)
    server.settimeout(None)
    return server, client


def test_distributed_import_does_not_import_cupy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kernelleaf.distributed; "
            "print(int('cupy' in sys.modules))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.stdout.strip() == "0"


@pytest.mark.parametrize("dtype", [
    np.bool_, np.int8, np.int32, np.int64, np.uint16,
    np.float16, np.float32, np.float64,
])
def test_json_ndarray_roundtrip_preserves_shape_dtype_and_values(dtype):
    if np.issubdtype(dtype, np.bool_):
        values = np.asarray([[True, False, True], [False, True, False]], dtype=dtype)
    else:
        values = np.arange(6, dtype=dtype).reshape(2, 3)
    decoded = JsonCodec().decode(JsonCodec().encode(_message(payload={
        "gradient": values,
        "nested": [{"bias": np.asarray(2, dtype=dtype)}],
    })))
    gradient = decoded.payload["gradient"]
    assert gradient.shape == values.shape
    assert gradient.dtype == values.dtype
    np.testing.assert_array_equal(gradient, values)
    scalar = decoded.payload["nested"][0]["bias"]
    assert scalar.shape == ()
    assert scalar.dtype == values.dtype


def test_json_codec_rejects_ambiguous_or_unsupported_arrays():
    codec = JsonCodec()
    with pytest.raises(CodecError, match="does not support dtype"):
        codec.encode(_message(payload={
            "values": np.asarray([1 + 2j], dtype=np.complex64)
        }))
    with pytest.raises(CodecError, match="finite"):
        codec.encode(_message(payload={
            "values": np.asarray([np.nan], dtype=np.float32)
        }))
    with pytest.raises(CodecError, match="reserved"):
        codec.encode(_message(payload={
            "value": {"__kernelleaf_ndarray__": False}
        }))


def test_message_schema_and_version_are_strict():
    assert {item.value for item in MessageType} == {
        "REGISTER", "PULL_PARAMETERS", "PUSH_GRADIENTS", "PARAMETERS",
        "HEARTBEAT", "METRICS", "SHUTDOWN", "ERROR",
    }
    with pytest.raises(ProtocolError, match="message_type"):
        Message("HEARTBEAT", "request", "worker", 0, {})
    value = _message().to_dict()
    value["protocol_version"] = 99
    with pytest.raises(CodecError, match="protocol version"):
        JsonCodec().decode(
            __import__("json").dumps(value).encode("utf-8")
        )
    value = _message().to_dict()
    value["extra"] = True
    with pytest.raises(CodecError, match="unknown fields"):
        JsonCodec().decode(
            __import__("json").dumps(value).encode("utf-8")
        )


def test_receive_frame_handles_fragmented_header_and_payload():
    receiver, sender = _tcp_pair()
    payload = JsonCodec().encode(_message(payload={
        "values": np.arange(17, dtype=np.float32)
    }))
    frame = pack_frame(payload)

    def fragmented_send():
        try:
            for start, end in ((0, 1), (1, 4), (4, 9), (9, len(frame))):
                sender.sendall(frame[start:end])
                time.sleep(0.005)
        finally:
            sender.close()

    thread = threading.Thread(target=fragmented_send)
    thread.start()
    try:
        assert receive_frame(receiver) == payload
    finally:
        receiver.close()
        thread.join(timeout=1)
    assert not thread.is_alive()


def test_multiple_frames_in_one_recv_are_not_merged():
    receiver, sender = _tcp_pair()
    codec = JsonCodec()
    first, second = _message(1), _message(2)
    sender.sendall(
        pack_frame(codec.encode(first)) + pack_frame(codec.encode(second))
    )
    transport = FramedTransport(receiver, codec)
    try:
        assert transport.receive() == first
        assert transport.receive() == second
    finally:
        receiver.close()
        sender.close()


@pytest.mark.parametrize("declared,max_size", [(0, 32), (33, 32)])
def test_invalid_frame_lengths_are_rejected(declared, max_size):
    receiver, sender = _tcp_pair()
    sender.sendall(struct.pack("!I", declared))
    try:
        with pytest.raises(InvalidFrameLength):
            receive_frame(receiver, max_frame_size=max_size)
    finally:
        receiver.close()
        sender.close()


def test_early_disconnect_reports_partial_frame():
    receiver, sender = _tcp_pair()
    sender.sendall(struct.pack("!I", 10) + b"abc")
    sender.close()
    try:
        with pytest.raises(PeerDisconnected, match="3 of 10"):
            receive_frame(receiver)
    finally:
        receiver.close()


def test_transport_timeout_is_explicit_and_restores_socket_timeout():
    receiver, sender = _tcp_pair()
    receiver.settimeout(None)
    transport = FramedTransport(receiver, JsonCodec(), timeout=0.03)
    try:
        with pytest.raises(TransportTimeout):
            transport.receive()
        assert receiver.gettimeout() is None
    finally:
        receiver.close()
        sender.close()


def test_transport_roundtrip_uses_codec_boundary_and_reports_bytes():
    left_socket, right_socket = _tcp_pair()
    left = FramedTransport(left_socket, JsonCodec())
    right = FramedTransport(right_socket, JsonCodec())
    message = _message(payload={"gradient": np.ones((2, 2), dtype=np.float32)})
    try:
        sent_bytes = left.send(message)
        assert sent_bytes > 4
        decoded = right.receive()
        assert decoded.request_id == message.request_id
        np.testing.assert_array_equal(
            decoded.payload["gradient"], message.payload["gradient"]
        )
    finally:
        left.close()
        right.close()


def test_transport_accepts_a_non_json_codec():
    class RequestCodec(MessageCodec):
        name = "request-id"

        def encode(self, message):
            return message.request_id.encode("ascii")

        def decode(self, payload):
            return Message(
                MessageType.HEARTBEAT, payload.decode("ascii"), "worker-0", 0, {}
            )

    left_socket, right_socket = _tcp_pair()
    left = FramedTransport(left_socket, RequestCodec())
    right = FramedTransport(right_socket, RequestCodec())
    try:
        left.send(_message(7))
        received = right.receive()
        assert received.message_type is MessageType.HEARTBEAT
        assert received.request_id == "request-7"
    finally:
        left.close()
        right.close()


def test_cupy_array_serializes_only_when_cuda_is_available():
    import kernelleaf as kl

    if not kl.is_cuda_available():
        pytest.skip("working CUDA/CuPy unavailable")
    cp = kl.cuda(0).xp
    values = cp.arange(6, dtype=cp.float32).reshape(2, 3)
    decoded = JsonCodec().decode(JsonCodec().encode(
        _message(payload={"values": values})
    ))
    assert isinstance(decoded.payload["values"], np.ndarray)
    np.testing.assert_array_equal(
        decoded.payload["values"], np.arange(6, dtype=np.float32).reshape(2, 3)
    )
