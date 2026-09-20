"""Launch a lightweight multi-node parameter server on one machine."""

from __future__ import annotations

import multiprocessing as mp
import socket
import time
from typing import Dict, List, Optional, Sequence

from .client import ParameterServerClient
from .protocol import VALID_MODES


def _free_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


class LocalCluster:
    """A parameter-server group emulated by independent OS processes."""

    def __init__(
        self,
        *,
        num_shards: int = 2,
        workers: int = 4,
        transport: str = "socket",
        mode: str = "sync",
        host: str = "127.0.0.1",
        barrier_timeout: float = 30.0,
        startup_timeout: float = 15.0,
        server_cpu_affinity: Optional[Sequence[Sequence[int]]] = None,
    ):
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if transport not in {"socket", "grpc"}:
            raise ValueError(f"unsupported transport: {transport!r}")
        if mode not in VALID_MODES:
            raise ValueError(f"unsupported consistency mode: {mode!r}")
        self.num_shards = int(num_shards)
        self.workers = int(workers)
        self.transport = str(transport)
        self.mode = str(mode)
        self.host = str(host)
        self.barrier_timeout = float(barrier_timeout)
        self.startup_timeout = float(startup_timeout)
        if server_cpu_affinity is not None:
            if len(server_cpu_affinity) < self.num_shards:
                raise ValueError(
                    "server_cpu_affinity must provide one CPU group per shard"
                )
            if any(not group for group in server_cpu_affinity[: self.num_shards]):
                raise ValueError("server CPU affinity groups must not be empty")
        self.server_cpu_affinity = (
            None
            if server_cpu_affinity is None
            else tuple(tuple(group) for group in server_cpu_affinity)
        )
        self._context = mp.get_context("spawn")
        self._processes: List[mp.Process] = []
        self._ports: List[int] = []
        self._started = False

    @property
    def endpoints(self) -> Sequence[str]:
        if not self._started:
            raise RuntimeError("cluster has not been started")
        return tuple(f"{self.host}:{port}" for port in self._ports)

    def start(self):
        if self._started:
            return self
        if self.transport == "grpc":
            from .grpc_transport import require_grpc

            require_grpc()
        if self.transport == "socket":
            from .socket_transport import run_socket_server

            target = run_socket_server
        else:
            from .grpc_transport import run_grpc_server

            target = run_grpc_server

        for shard_index in range(self.num_shards):
            port = _free_tcp_port(self.host)
            ready = self._context.Event()
            process = self._context.Process(
                target=target,
                kwargs={
                    "host": self.host,
                    "port": port,
                    "shard_index": shard_index,
                    "num_shards": self.num_shards,
                    "mode": self.mode,
                    "barrier_timeout": self.barrier_timeout,
                    "ready_event": ready,
                    "cpu_affinity": (
                        None
                        if self.server_cpu_affinity is None
                        else self.server_cpu_affinity[shard_index]
                    ),
                },
                name=f"kernelleaf-ps-{shard_index}",
                daemon=True,
            )
            process.start()
            self._processes.append(process)
            self._ports.append(port)
            deadline = time.monotonic() + self.startup_timeout
            while not ready.is_set():
                if process.exitcode is not None:
                    self.close()
                    raise RuntimeError(
                        f"parameter-server shard {shard_index} exited with "
                        f"code {process.exitcode}"
                    )
                if time.monotonic() >= deadline:
                    self.close()
                    raise TimeoutError(
                        f"parameter-server shard {shard_index} did not become ready"
                    )
                time.sleep(0.02)
        self._started = True
        return self

    def set_mode(self, mode: str) -> List[Dict[str, object]]:
        if mode not in VALID_MODES:
            raise ValueError(f"unsupported consistency mode: {mode!r}")
        client = ParameterServerClient(
            self.endpoints,
            transport=self.transport,
            timeout=self.barrier_timeout,
        )
        try:
            results = client.set_mode(mode)
        finally:
            client.close()
        self.mode = str(mode)
        return results

    def close(self):
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=5.0)
        self._processes = []
        self._ports = []
        self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
