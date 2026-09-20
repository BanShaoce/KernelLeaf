"""Length-prefixed TCP framing independent from message serialization."""

from __future__ import annotations

import socket
import struct
from typing import Optional

from .protocol import Message
from .serialization import MessageCodec


HEADER_SIZE = 4
DEFAULT_MAX_FRAME_SIZE = 64 * 1024 * 1024
_HEADER = struct.Struct("!I")


class TransportError(ConnectionError):
    """Base class for framed transport failures."""


class PeerDisconnected(TransportError):
    """Raised when a peer closes before a complete frame arrives."""


class InvalidFrameLength(TransportError):
    """Raised for empty or unreasonably large frames."""


class TransportTimeout(TransportError, TimeoutError):
    """Raised when socket I/O exceeds its configured timeout."""


def _validate_max_frame_size(max_frame_size):
    if isinstance(max_frame_size, bool) or not isinstance(max_frame_size, int) \
            or not 0 < max_frame_size <= 0xFFFFFFFF:
        raise ValueError("max_frame_size must be in [1, 2**32 - 1]")


def pack_frame(payload, max_frame_size=DEFAULT_MAX_FRAME_SIZE):
    """Return one length-prefixed frame without knowing its codec."""
    _validate_max_frame_size(max_frame_size)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("frame payload must be bytes-like")
    payload = bytes(payload)
    if not payload or len(payload) > max_frame_size:
        raise InvalidFrameLength(
            f"frame length {len(payload)} is outside [1, {max_frame_size}]"
        )
    return _HEADER.pack(len(payload)) + payload


def recv_exact(sock, size):
    """Receive exactly ``size`` bytes, handling arbitrary TCP fragmentation."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")
    chunks = bytearray()
    while len(chunks) < size:
        try:
            chunk = sock.recv(size - len(chunks))
        except socket.timeout as error:
            raise TransportTimeout(
                f"timed out after receiving {len(chunks)} of {size} bytes"
            ) from error
        except OSError as error:
            raise TransportError(f"socket receive failed: {error}") from error
        if not chunk:
            raise PeerDisconnected(
                f"peer disconnected after {len(chunks)} of {size} bytes"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def receive_frame(sock, max_frame_size=DEFAULT_MAX_FRAME_SIZE):
    _validate_max_frame_size(max_frame_size)
    header = recv_exact(sock, HEADER_SIZE)
    length = _HEADER.unpack(header)[0]
    if length == 0 or length > max_frame_size:
        raise InvalidFrameLength(
            f"received frame length {length} outside [1, {max_frame_size}]"
        )
    return recv_exact(sock, length)


def send_frame(sock, payload, max_frame_size=DEFAULT_MAX_FRAME_SIZE):
    frame = pack_frame(payload, max_frame_size=max_frame_size)
    try:
        sock.sendall(frame)
    except socket.timeout as error:
        raise TransportTimeout("timed out while sending frame") from error
    except OSError as error:
        raise TransportError(f"socket send failed: {error}") from error
    return len(frame)


class FramedTransport:
    """A connected socket paired with any ``MessageCodec`` implementation."""

    def __init__(self, sock, codec, *, max_frame_size=DEFAULT_MAX_FRAME_SIZE,
                 timeout: Optional[float] = None):
        if not isinstance(codec, MessageCodec):
            raise TypeError("codec must implement MessageCodec")
        _validate_max_frame_size(max_frame_size)
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")
        self.socket = sock
        self.codec = codec
        self.max_frame_size = max_frame_size
        self.timeout = timeout

    def _set_timeout(self):
        previous = self.socket.gettimeout()
        if self.timeout is not None:
            self.socket.settimeout(self.timeout)
        return previous

    def send(self, message: Message):
        previous = self._set_timeout()
        try:
            return send_frame(
                self.socket, self.codec.encode(message), self.max_frame_size
            )
        finally:
            if self.timeout is not None:
                self.socket.settimeout(previous)

    def receive(self) -> Message:
        previous = self._set_timeout()
        try:
            payload = receive_frame(self.socket, self.max_frame_size)
        finally:
            if self.timeout is not None:
                self.socket.settimeout(previous)
        return self.codec.decode(payload)

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
