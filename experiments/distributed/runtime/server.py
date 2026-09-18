"""Thread-safe sharded parameter-server state and request dispatch."""

from __future__ import annotations

import threading
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .protocol import (
    ParameterEntry,
    ParameterServerError,
    Request,
    Response,
    VALID_MODES,
    merge_entries,
)


class ParameterServerState:
    """One parameter shard used by a socket or gRPC server node.

    Sparse range operations are represented by dictionaries keyed by arbitrary
    strings. Sync mode installs a per-clock barrier across all registered
    workers. Async mode applies each push immediately.
    """

    def __init__(
        self,
        shard_index: int = 0,
        num_shards: int = 1,
        *,
        mode: str = "sync",
        barrier_timeout: float = 30.0,
        initial_values: Optional[Mapping[str, Sequence[float] | float]] = None,
    ):
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if not 0 <= shard_index < num_shards:
            raise ValueError("shard_index must be in [0, num_shards)")
        if mode not in VALID_MODES:
            raise ValueError(f"unsupported consistency mode: {mode!r}")
        if barrier_timeout <= 0:
            raise ValueError("barrier_timeout must be positive")
        self.shard_index = int(shard_index)
        self.num_shards = int(num_shards)
        self.barrier_timeout = float(barrier_timeout)
        self._condition = threading.Condition(threading.RLock())
        self._mode = mode
        self._values: Dict[str, Tuple[float, ...]] = {}
        self._versions: Dict[str, int] = {}
        self._workers = set()
        self._worker_modes: Dict[str, str] = {}
        self._pending: Dict[str, Dict[int, Dict[str, ParameterEntry]]] = {}
        self._applied_clock: Dict[str, int] = {}
        self._global_clock = 0
        if initial_values:
            for key, value in initial_values.items():
                values = (float(value),) if isinstance(value, (int, float)) else tuple(
                    float(item) for item in value
                )
                self._values[str(key)] = values
                self._versions[str(key)] = 1

    @property
    def mode(self) -> str:
        with self._condition:
            return self._mode

    def register_worker(self, worker_id: str) -> Dict[str, object]:
        worker_id = str(worker_id)
        if not worker_id:
            raise ParameterServerError("worker_id must not be empty")
        with self._condition:
            self._workers.add(worker_id)
            self._worker_modes[worker_id] = self._mode
            self._pending.setdefault(worker_id, {})
            self._applied_clock.setdefault(worker_id, -1)
            self._condition.notify_all()
            return self.metadata_locked()

    def pull(self, keys: Iterable[str]) -> Tuple[Tuple[ParameterEntry, ...], Dict[str, int]]:
        requested = tuple(dict.fromkeys(str(key) for key in keys))
        with self._condition:
            values = []
            versions = {}
            for key in requested:
                values.append(ParameterEntry(key, self._values.get(key, (0.0,)), "set"))
                versions[key] = self._versions.get(key, 0)
            return tuple(values), versions

    def push(
        self,
        worker_id: str,
        clock: int,
        entries: Iterable[ParameterEntry],
    ) -> Dict[str, object]:
        worker_id = str(worker_id)
        if not worker_id:
            raise ParameterServerError("worker_id must not be empty")
        clock = int(clock)
        if clock < 0:
            raise ParameterServerError("clock must be non-negative")
        normalized = merge_entries(entries)
        with self._condition:
            if worker_id not in self._workers:
                self._workers.add(worker_id)
                self._worker_modes[worker_id] = self._mode
                self._pending.setdefault(worker_id, {})
                self._applied_clock.setdefault(worker_id, -1)

            worker_mode = self._worker_modes.get(worker_id, self._mode)
            if self._mode == "async" or worker_mode == "async":
                applied = self._apply_locked(normalized)
                self._applied_clock[worker_id] = max(
                    self._applied_clock.get(worker_id, -1), clock
                )
                return {
                    "status": "applied",
                    "applied": applied,
                    "version": self._global_clock,
                    "mode": self._mode,
                }

            worker_pending = self._pending.setdefault(worker_id, {})
            current = dict(worker_pending.get(clock, {}))
            for entry in normalized:
                key = entry.key
                if key not in current:
                    current[key] = entry
                else:
                    current[key] = merge_entries((current[key], entry))[0]
            worker_pending[clock] = current
            deadline = time.monotonic() + self.barrier_timeout

            while True:
                if self._mode == "async":
                    return {
                        "status": "switched",
                        "applied": 0,
                        "version": self._global_clock,
                        "mode": self._mode,
                    }
                if (
                    self._applied_clock.get(worker_id, -1) >= clock
                    and clock not in self._pending.get(worker_id, {})
                ):
                    return {
                        "status": "applied",
                        "applied": 0,
                        "version": self._global_clock,
                        "mode": self._mode,
                        "joined": True,
                    }
                ready = all(
                    clock in self._pending.get(other, {})
                    for other in self._workers
                    if self._worker_modes.get(other, self._mode) == "sync"
                )
                if ready:
                    grouped = []
                    for other in sorted(self._workers):
                        if self._worker_modes.get(other, self._mode) != "sync":
                            continue
                        grouped.extend(self._pending[other].pop(clock).values())
                    applied = self._apply_locked(merge_entries(grouped))
                    for other in self._workers:
                        if self._worker_modes.get(other, self._mode) != "sync":
                            continue
                        self._applied_clock[other] = max(
                            self._applied_clock.get(other, -1), clock
                        )
                    self._condition.notify_all()
                    return {
                        "status": "applied",
                        "applied": applied,
                        "version": self._global_clock,
                        "mode": self._mode,
                        "workers": len(self._workers),
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ParameterServerError(
                        f"sync barrier timed out at clock {clock} on shard "
                        f"{self.shard_index}"
                    )
                self._condition.wait(remaining)

    def set_mode(self, mode: str, worker_id: str = "") -> Dict[str, object]:
        if mode not in VALID_MODES:
            raise ParameterServerError(f"unsupported consistency mode: {mode!r}")
        with self._condition:
            previous = self._mode
            if mode == previous:
                if worker_id:
                    self._worker_modes[str(worker_id)] = mode
                return {"previous": previous, "mode": mode, "flushed": 0}
            self._mode = mode
            if worker_id:
                self._worker_modes[str(worker_id)] = mode
            else:
                for registered in self._workers:
                    self._worker_modes[registered] = mode
            flushed = 0
            if mode == "async":
                for worker_id in sorted(self._pending):
                    clocks = sorted(self._pending[worker_id])
                    for clock in clocks:
                        pending = self._pending[worker_id].pop(clock)
                        if pending:
                            flushed += self._apply_locked(tuple(pending.values()))
                        self._applied_clock[worker_id] = max(
                            self._applied_clock.get(worker_id, -1), clock
                        )
            self._condition.notify_all()
            return {
                "previous": previous,
                "mode": mode,
                "flushed": flushed,
                "version": self._global_clock,
            }

    def metadata_locked(self) -> Dict[str, object]:
        return {
            "mode": self._mode,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "num_parameters": len(self._values),
            "num_workers": len(self._workers),
            "workers": sorted(self._workers),
            "worker_modes": dict(self._worker_modes),
            "global_clock": self._global_clock,
            "applied_clock": dict(self._applied_clock),
        }

    def metadata(self) -> Dict[str, object]:
        with self._condition:
            return self.metadata_locked()

    def _apply_locked(self, entries: Iterable[ParameterEntry]) -> int:
        applied = 0
        for entry in entries:
            if entry.op == "set" or entry.key not in self._values:
                self._values[entry.key] = entry.values
            else:
                previous = self._values[entry.key]
                if len(previous) != len(entry.values):
                    raise ParameterServerError(
                        f"value-length mismatch for key {entry.key!r}: "
                        f"{len(previous)} != {len(entry.values)}"
                    )
                self._values[entry.key] = tuple(
                    left + right for left, right in zip(previous, entry.values)
                )
            self._versions[entry.key] = self._versions.get(entry.key, 0) + 1
            self._global_clock += 1
            applied += 1
        return applied


def dispatch_request(state: ParameterServerState, request: Request) -> Response:
    """Execute a transport-neutral request against one parameter shard."""
    try:
        if request.action == "ping":
            return Response(version=int(state.metadata()["global_clock"]))
        if request.action == "register":
            metadata = state.register_worker(request.worker_id)
            return Response(version=int(metadata["global_clock"]), metadata=metadata)
        if request.action == "pull":
            values, versions = state.pull(request.keys)
            metadata = state.metadata()
            metadata["versions"] = versions
            return Response(
                version=int(metadata["global_clock"]),
                values=values,
                metadata=metadata,
            )
        if request.action == "push":
            metadata = state.push(request.worker_id, request.clock, request.entries)
            metadata["server"] = state.metadata()
            return Response(version=int(metadata["version"]), metadata=metadata)
        if request.action == "set_mode":
            metadata = state.set_mode(request.mode, request.worker_id)
            metadata["server"] = state.metadata()
            return Response(
                version=int(metadata.get("version", metadata["server"]["global_clock"])),
                metadata=metadata,
            )
        if request.action == "metadata":
            metadata = state.metadata()
            return Response(version=int(metadata["global_clock"]), metadata=metadata)
        raise ParameterServerError(f"unknown request action: {request.action!r}")
    except Exception as error:  # The response is part of the wire protocol.
        return Response(ok=False, error=f"{type(error).__name__}: {error}")
