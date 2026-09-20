"""Transport clients used by the distributed parameter-server layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
import threading
from typing import Tuple

from .protocol import Request, Response


class TransportClient(ABC):
    """A request/response channel to one parameter-server shard."""

    def __init__(self):
        self._metrics_lock = threading.Lock()
        self._bytes_sent = 0
        self._bytes_received = 0
        self._requests = 0

    def _record_exchange(self, request_bytes: int, response_bytes: int) -> None:
        with self._metrics_lock:
            self._bytes_sent += int(request_bytes)
            self._bytes_received += int(response_bytes)
            self._requests += 1

    def metrics(self):
        """Return payload bytes counted before network framing overhead."""
        with self._metrics_lock:
            return {
                "requests": int(self._requests),
                "bytes_sent": int(self._bytes_sent),
                "bytes_received": int(self._bytes_received),
                "message_bytes": int(self._bytes_sent + self._bytes_received),
            }

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self._bytes_sent = 0
            self._bytes_received = 0
            self._requests = 0

    @abstractmethod
    def request(self, request: Request) -> Response:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def parse_endpoint(endpoint: str) -> Tuple[str, int]:
    """Parse a ``host:port`` endpoint without accepting implicit hosts."""
    text = str(endpoint).strip()
    if not text:
        raise ValueError("endpoint must not be empty")
    host, separator, port_text = text.rpartition(":")
    if not separator or not host:
        raise ValueError(f"endpoint must use host:port syntax: {endpoint!r}")
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError(f"invalid endpoint port: {endpoint!r}") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"endpoint port is out of range: {endpoint!r}")
    return host, port


def create_transport_client(
    endpoint: str, transport: str = "socket", *, timeout: float = 30.0
) -> TransportClient:
    """Create a transport client while keeping optional dependencies lazy."""
    if transport == "socket":
        from .socket_transport import SocketTransportClient

        return SocketTransportClient(endpoint, timeout=timeout)
    if transport == "grpc":
        from .grpc_transport import GRPCTransportClient

        return GRPCTransportClient(endpoint, timeout=timeout)
    raise ValueError(f"unsupported parameter-server transport: {transport!r}")
