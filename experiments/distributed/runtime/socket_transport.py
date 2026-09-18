"""Line-delimited JSON transport for the parameter server."""

from __future__ import annotations

import json
import socket
import socketserver
import threading
from typing import Optional

from .cpu_affinity import set_current_process_affinity
from .protocol import Request, Response
from .server import ParameterServerState, dispatch_request
from .transport import TransportClient, parse_endpoint


MAX_REQUEST_BYTES = 16 * 1024 * 1024


class SocketTransportClient(TransportClient):
    """Persistent TCP connection using newline-delimited JSON messages."""

    def __init__(self, endpoint: str, *, timeout: float = 30.0):
        super().__init__()
        self.endpoint = str(endpoint)
        self.host, self.port = parse_endpoint(self.endpoint)
        self.timeout = float(timeout)
        self._socket: Optional[socket.socket] = None
        self._reader = None
        self._lock = threading.Lock()

    def _connect(self):
        self.close()
        connection = socket.create_connection((self.host, self.port), self.timeout)
        connection.settimeout(self.timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket = connection
        self._reader = connection.makefile("rb")

    def _request_once(self, request: Request) -> Response:
        if self._socket is None:
            self._connect()
        wire = request.to_json().encode("utf-8") + b"\n"
        self._socket.sendall(wire)
        line = self._reader.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            raise ConnectionError(f"parameter server closed {self.endpoint}")
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            raise ValueError("parameter-server response exceeds the size limit")
        response = Response.from_json(line.decode("utf-8"))
        self._record_exchange(len(wire), len(line))
        return response

    def request(self, request: Request) -> Response:
        with self._lock:
            try:
                return self._request_once(request)
            except (ConnectionError, OSError, json.JSONDecodeError):
                self.close()
                if request.action == "push":
                    raise
                return self._request_once(request)

    def close(self):
        if self._reader is not None:
            try:
                self._reader.close()
            except OSError:
                pass
            self._reader = None
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _SocketRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            while True:
                line = self.rfile.readline(MAX_REQUEST_BYTES + 1)
                if not line:
                    return
                if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
                    response = Response(
                        ok=False, error="ValueError: request exceeds the size limit"
                    )
                else:
                    try:
                        request = Request.from_json(line.decode("utf-8"))
                        response = dispatch_request(self.server.state, request)
                    except Exception as error:
                        response = Response(
                            ok=False, error=f"{type(error).__name__}: {error}"
                        )
                self.wfile.write(response.to_json().encode("utf-8") + b"\n")
                self.wfile.flush()
        except (ConnectionError, OSError):
            return


class SocketServerHandle:
    """In-process threaded socket server useful for tests and embedding."""

    def __init__(self, host: str, port: int, state: ParameterServerState):
        self._server = _ThreadingTCPServer((host, int(port)), _SocketRequestHandler)
        self._server.state = state
        self._thread: Optional[threading.Thread] = None

    @property
    def host(self) -> str:
        return self._server.server_address[0]

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"kernelleaf-psserver-{self.port}",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self):
        if self._thread is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        self._thread = None


def run_socket_server(
    host: str,
    port: int,
    *,
    shard_index: int,
    num_shards: int,
    mode: str,
    barrier_timeout: float,
    ready_event,
    cpu_affinity=None,
):
    """Process entry point for one socket parameter-server shard."""
    set_current_process_affinity(cpu_affinity)
    state = ParameterServerState(
        shard_index,
        num_shards,
        mode=mode,
        barrier_timeout=barrier_timeout,
    )
    server = _ThreadingTCPServer((host, int(port)), _SocketRequestHandler)
    server.state = state
    ready_event.set()
    try:
        server.serve_forever()
    finally:
        server.server_close()
