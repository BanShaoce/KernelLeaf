"""Wire-level messages shared by all parameter-server transports."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


VALID_MODES = ("sync", "async")
VALID_OPS = ("add", "set")


class ParameterServerError(RuntimeError):
    """Raised when a parameter-server node rejects a request."""


@dataclass(frozen=True)
class ParameterEntry:
    """A sparse parameter update or a value returned by a pull."""

    key: str
    values: Tuple[float, ...]
    op: str = "add"

    def __post_init__(self):
        key = str(self.key)
        if not key:
            raise ValueError("parameter key must not be empty")
        op = str(self.op)
        if op not in VALID_OPS:
            raise ValueError(f"unsupported parameter operation: {op!r}")
        values = tuple(float(value) for value in self.values)
        if not values:
            raise ValueError("parameter entry must contain at least one value")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "values", values)

    def to_json(self) -> Dict[str, Any]:
        return {"key": self.key, "values": list(self.values), "op": self.op}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ParameterEntry":
        return cls(
            key=value["key"],
            values=value.get("values", ()),
            op=value.get("op", "add"),
        )


@dataclass(frozen=True)
class Request:
    """Transport-independent parameter-server request."""

    action: str
    worker_id: str = ""
    clock: int = 0
    keys: Tuple[str, ...] = ()
    entries: Tuple[ParameterEntry, ...] = ()
    mode: str = ""

    def __post_init__(self):
        action = str(self.action)
        if not action:
            raise ValueError("request action must not be empty")
        clock = int(self.clock)
        if clock < 0:
            raise ValueError("worker clock must be non-negative")
        mode = str(self.mode)
        if mode and mode not in VALID_MODES:
            raise ValueError(f"unsupported consistency mode: {mode!r}")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "worker_id", str(self.worker_id))
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "keys", tuple(str(key) for key in self.keys))
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "mode", mode)

    def to_json(self) -> str:
        return json.dumps(
            {
                "action": self.action,
                "worker_id": self.worker_id,
                "clock": self.clock,
                "keys": list(self.keys),
                "entries": [entry.to_json() for entry in self.entries],
                "mode": self.mode,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, data: str) -> "Request":
        value = json.loads(data)
        return cls(
            action=value["action"],
            worker_id=value.get("worker_id", ""),
            clock=value.get("clock", 0),
            keys=value.get("keys", ()),
            entries=tuple(
                ParameterEntry.from_json(item) for item in value.get("entries", ())
            ),
            mode=value.get("mode", ""),
        )


@dataclass(frozen=True)
class Response:
    """Transport-independent parameter-server response."""

    ok: bool = True
    error: str = ""
    version: int = 0
    values: Tuple[ParameterEntry, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": bool(self.ok),
                "error": self.error,
                "version": int(self.version),
                "values": [entry.to_json() for entry in self.values],
                "metadata": self.metadata,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, data: str) -> "Response":
        value = json.loads(data)
        return cls(
            ok=bool(value.get("ok", False)),
            error=str(value.get("error", "")),
            version=int(value.get("version", 0)),
            values=tuple(
                ParameterEntry.from_json(item) for item in value.get("values", ())
            ),
            metadata=dict(value.get("metadata", {})),
        )


def entries_from_mapping(
    values: Mapping[str, Sequence[float] | float], op: str = "add"
) -> Tuple[ParameterEntry, ...]:
    """Normalize a Python mapping into sparse parameter entries."""
    entries = []
    for key, value in values.items():
        if isinstance(value, (int, float)):
            normalized: Iterable[float] = (float(value),)
        else:
            normalized = value
        entries.append(ParameterEntry(str(key), tuple(normalized), op=op))
    return tuple(entries)


def merge_entries(
    entries: Iterable[ParameterEntry],
) -> Tuple[ParameterEntry, ...]:
    """Merge entries by key while preserving add/set semantics."""
    merged: Dict[str, ParameterEntry] = {}
    for entry in entries:
        previous = merged.get(entry.key)
        if previous is None:
            merged[entry.key] = entry
            continue
        if previous.op == "set" or entry.op == "set":
            if previous.op == entry.op == "set":
                merged[entry.key] = entry
            else:
                raise ValueError(
                    f"cannot merge add and set updates for key {entry.key!r}"
                )
            continue
        if len(previous.values) != len(entry.values):
            raise ValueError(
                f"value-length mismatch for key {entry.key!r}: "
                f"{len(previous.values)} != {len(entry.values)}"
            )
        merged[entry.key] = ParameterEntry(
            previous.key,
            tuple(a + b for a, b in zip(previous.values, entry.values)),
            op="add",
        )
    return tuple(merged.values())
