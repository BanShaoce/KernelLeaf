"""Socket communication primitives for future distributed training layers."""

from .protocol import (
    PROTOCOL_VERSION,
    Message,
    MessageType,
    ProtocolError,
    UnsupportedProtocolVersion,
)
from .serialization import CodecError, JsonCodec, MessageCodec
from .parameter_server import (
    ParameterServer,
    ParameterServerError,
    ParameterSpec,
    parameter_schema,
    stable_named_parameters,
)
from .transport import (
    DEFAULT_MAX_FRAME_SIZE,
    FramedTransport,
    InvalidFrameLength,
    PeerDisconnected,
    TransportError,
    TransportTimeout,
    pack_frame,
    receive_frame,
    recv_exact,
    send_frame,
)
from .worker import Worker, WorkerProtocolError
from .launcher import (
    DistributedProcessError, LaunchResult, Launcher, RoleSpec,
)
from .monitor import (
    METRIC_FIELDS, JsonlMonitor, MetricValidationError, validate_metric,
)

__all__ = [
    "PROTOCOL_VERSION", "Message", "MessageType", "ProtocolError",
    "UnsupportedProtocolVersion", "CodecError", "JsonCodec", "MessageCodec",
    "DEFAULT_MAX_FRAME_SIZE", "FramedTransport", "InvalidFrameLength",
    "PeerDisconnected", "TransportError", "TransportTimeout", "pack_frame",
    "receive_frame", "recv_exact", "send_frame",
    "ParameterServer", "ParameterServerError", "ParameterSpec",
    "parameter_schema", "stable_named_parameters", "Worker",
    "WorkerProtocolError",
    "DistributedProcessError", "LaunchResult", "Launcher", "RoleSpec",
    "METRIC_FIELDS", "JsonlMonitor", "MetricValidationError",
    "validate_metric",
]
