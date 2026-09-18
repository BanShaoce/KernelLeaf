"""Sharded parameter-server client with selectable consistency."""

from __future__ import annotations

import threading
import zlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .protocol import (
    ParameterEntry,
    ParameterServerError,
    Request,
    Response,
    VALID_MODES,
    entries_from_mapping,
)
from .transport import TransportClient, create_transport_client


def shard_for_key(key: str, num_shards: int) -> int:
    """Assign a sparse parameter key to a stable shard."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    checksum = zlib.crc32(str(key).encode("utf-8")) & 0xFFFFFFFF
    return checksum % int(num_shards)


class ParameterServerClient:
    """Client for a sharded parameter server.

    ``pull`` and ``push`` accept arbitrary sparse keys. A push is sent to every
    shard, including an empty heartbeat, so sync barriers cannot be skipped when
    a worker has no update for a particular range.
    """

    def __init__(
        self,
        endpoints: Sequence[str],
        *,
        worker_id: str = "",
        transport: str = "socket",
        timeout: float = 30.0,
    ):
        if not endpoints:
            raise ValueError("at least one parameter-server endpoint is required")
        self.endpoints = tuple(str(endpoint) for endpoint in endpoints)
        self.worker_id = str(worker_id)
        self.transport = str(transport)
        self.timeout = float(timeout)
        self._clients: List[TransportClient] = [
            create_transport_client(endpoint, transport, timeout=timeout)
            for endpoint in self.endpoints
        ]
        self._lock = threading.Lock()

    @property
    def num_shards(self) -> int:
        return len(self._clients)

    def _request(self, shard: int, request: Request) -> Response:
        response = self._clients[shard].request(request)
        if not response.ok:
            raise ParameterServerError(response.error)
        return response

    def register(self, worker_id: Optional[str] = None) -> List[Dict[str, object]]:
        if worker_id is not None:
            self.worker_id = str(worker_id)
        if not self.worker_id:
            raise ValueError("worker_id must be set before registration")
        with self._lock:
            return [
                dict(
                    self._request(
                        shard,
                        Request(
                            "register",
                            worker_id=self.worker_id,
                            clock=0,
                        ),
                    ).metadata
                )
                for shard in range(self.num_shards)
            ]

    def pull(self, keys: Iterable[str]) -> Dict[str, Tuple[float, ...]]:
        grouped: Dict[int, List[str]] = {shard: [] for shard in range(self.num_shards)}
        for key in dict.fromkeys(str(key) for key in keys):
            grouped[shard_for_key(key, self.num_shards)].append(key)
        values: Dict[str, Tuple[float, ...]] = {}
        with self._lock:
            for shard, shard_keys in grouped.items():
                if not shard_keys:
                    continue
                response = self._request(
                    shard,
                    Request("pull", worker_id=self.worker_id, keys=tuple(shard_keys)),
                )
                values.update({entry.key: entry.values for entry in response.values})
        return values

    def pull_scalars(self, keys: Iterable[str]) -> Dict[str, float]:
        pulled = self.pull(keys)
        return {key: values[0] for key, values in pulled.items()}

    def push(
        self,
        updates: Mapping[str, Sequence[float] | float] | Iterable[ParameterEntry],
        *,
        clock: int,
        op: str = "add",
        heartbeat: bool = True,
    ) -> List[Dict[str, object]]:
        if isinstance(updates, Mapping):
            entries = entries_from_mapping(updates, op=op)
        else:
            entries = tuple(updates)
            if op != "add":
                entries = tuple(
                    ParameterEntry(entry.key, entry.values, op=op) for entry in entries
                )
        grouped: Dict[int, List[ParameterEntry]] = {
            shard: [] for shard in range(self.num_shards)
        }
        for entry in entries:
            grouped[shard_for_key(entry.key, self.num_shards)].append(entry)
        responses = []
        with self._lock:
            for shard in range(self.num_shards):
                if not grouped[shard] and not heartbeat:
                    continue
                response = self._request(
                    shard,
                    Request(
                        "push",
                        worker_id=self.worker_id,
                        clock=clock,
                        entries=tuple(grouped[shard]),
                    ),
                )
                responses.append(dict(response.metadata))
        return responses

    def barrier(self, clock: int) -> List[Dict[str, object]]:
        """Enter the next synchronous round without sending a parameter update."""
        return self.push({}, clock=clock, heartbeat=True)

    def set_mode(self, mode: str) -> List[Dict[str, object]]:
        if mode not in VALID_MODES:
            raise ValueError(f"unsupported consistency mode: {mode!r}")
        with self._lock:
            return [
                dict(
                    self._request(
                        shard,
                        Request("set_mode", mode=mode, worker_id=self.worker_id),
                    ).metadata
                )
                for shard in range(self.num_shards)
            ]

    def metadata(self) -> List[Dict[str, object]]:
        with self._lock:
            return [
                dict(self._request(shard, Request("metadata")).metadata)
                for shard in range(self.num_shards)
            ]

    def transport_metrics(self) -> Dict[str, int]:
        """Aggregate serialized request/response bytes across all shards."""
        metrics = {
            "requests": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "message_bytes": 0,
        }
        for client in self._clients:
            shard_metrics = client.metrics()
            for key in metrics:
                metrics[key] += int(shard_metrics.get(key, 0))
        return metrics

    def reset_transport_metrics(self) -> None:
        for client in self._clients:
            client.reset_metrics()

    def close(self):
        with self._lock:
            for client in self._clients:
                client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
