"""Versioned messages shared by KernelLeaf distributed transports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping


PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """Raised when a distributed message violates the wire protocol."""


class UnsupportedProtocolVersion(ProtocolError):
    """Raised when peers do not speak the same protocol version."""


class MessageType(str, Enum):
    REGISTER = "REGISTER"
    PULL_PARAMETERS = "PULL_PARAMETERS"
    PUSH_GRADIENTS = "PUSH_GRADIENTS"
    PARAMETERS = "PARAMETERS"
    HEARTBEAT = "HEARTBEAT"
    METRICS = "METRICS"
    SHUTDOWN = "SHUTDOWN"
    ERROR = "ERROR"


_MESSAGE_FIELDS = {
    "protocol_version", "message_type", "request_id", "worker_id", "step",
    "payload",
}


@dataclass(frozen=True)
class Message:
    """One validated V12.1 protocol message.

    ``request_id`` correlates a response with its request. ``worker_id`` names
    the sender (servers should use a stable server identifier), and ``step`` is
    the sender's non-negative parameter/training version.
    """

    message_type: MessageType
    request_id: str
    worker_id: str
    step: int
    payload: Mapping[str, Any]
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self):
        if (isinstance(self.protocol_version, bool)
                or not isinstance(self.protocol_version, int)
                or self.protocol_version != PROTOCOL_VERSION):
            raise UnsupportedProtocolVersion(
                f"unsupported protocol version {self.protocol_version}; "
                f"expected {PROTOCOL_VERSION}"
            )
        if not isinstance(self.message_type, MessageType):
            raise ProtocolError("message_type must be a MessageType")
        for name, value in (
            ("request_id", self.request_id), ("worker_id", self.worker_id)
        ):
            if not isinstance(value, str) or not value:
                raise ProtocolError(f"{name} must be a non-empty string")
        if isinstance(self.step, bool) or not isinstance(self.step, int) \
                or self.step < 0:
            raise ProtocolError("step must be a non-negative integer")
        if not isinstance(self.payload, Mapping):
            raise ProtocolError("payload must be a mapping")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "message_type": self.message_type.value,
            "request_id": self.request_id,
            "worker_id": self.worker_id,
            "step": self.step,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Message":
        if not isinstance(value, Mapping):
            raise ProtocolError("decoded message must be a mapping")
        fields = set(value)
        missing = sorted(_MESSAGE_FIELDS - fields)
        unknown = sorted(fields - _MESSAGE_FIELDS)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing fields: {missing}")
            if unknown:
                details.append(f"unknown fields: {unknown}")
            raise ProtocolError("invalid message schema (" + "; ".join(details) + ")")
        try:
            message_type = MessageType(value["message_type"])
        except (TypeError, ValueError) as error:
            raise ProtocolError(
                f"unknown message type: {value['message_type']!r}"
            ) from error
        return cls(
            protocol_version=value["protocol_version"],
            message_type=message_type,
            request_id=value["request_id"],
            worker_id=value["worker_id"],
            step=value["step"],
            payload=value["payload"],
        )
