"""Optional gRPC + protobuf transport for the parameter server."""

from __future__ import annotations

from concurrent import futures
import json
import threading
from typing import Optional

from .cpu_affinity import set_current_process_affinity
from .protocol import ParameterEntry, Request, Response
from .server import ParameterServerState, dispatch_request
from .transport import TransportClient


_RUNTIME = None


def require_grpc():
    """Load grpcio lazily and build the protobuf descriptors once."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    try:
        import grpc
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    except ImportError as error:
        raise RuntimeError(
            "gRPC transport requires optional dependencies. Install with "
            "`pip install -e '.[grpc]'`."
        ) from error

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "kernelleaf.proto"
    file_proto.package = "kernelleaf.distributed"
    file_proto.syntax = "proto3"

    entry = file_proto.message_type.add()
    entry.name = "ParameterEntry"
    _field(entry, "key", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(
        entry,
        "values",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
        repeated=True,
    )
    _field(entry, "op", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    request = file_proto.message_type.add()
    request.name = "ParameterRequest"
    _field(request, "action", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(request, "worker_id", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(request, "clock", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _field(
        request,
        "keys",
        4,
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        repeated=True,
    )
    _field(
        request,
        "entries",
        5,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        repeated=True,
        type_name=".kernelleaf.distributed.ParameterEntry",
    )
    _field(request, "mode", 6, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    response = file_proto.message_type.add()
    response.name = "ParameterResponse"
    _field(response, "ok", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)
    _field(response, "error", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _field(response, "version", 3, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    _field(
        response,
        "values",
        4,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        repeated=True,
        type_name=".kernelleaf.distributed.ParameterEntry",
    )
    _field(
        response,
        "metadata_json",
        5,
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
    )

    service = file_proto.service.add()
    service.name = "ParameterServer"
    method = service.method.add()
    method.name = "Request"
    method.input_type = ".kernelleaf.distributed.ParameterRequest"
    method.output_type = ".kernelleaf.distributed.ParameterResponse"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    request_descriptor = pool.FindMessageTypeByName(
        "kernelleaf.distributed.ParameterRequest"
    )
    response_descriptor = pool.FindMessageTypeByName(
        "kernelleaf.distributed.ParameterResponse"
    )
    request_class = _message_class(message_factory, request_descriptor)
    response_class = _message_class(message_factory, response_descriptor)
    _RUNTIME = {
        "grpc": grpc,
        "request_class": request_class,
        "response_class": response_class,
    }
    return _RUNTIME


def _field(
    message,
    name: str,
    number: int,
    field_type: int,
    *,
    repeated: bool = False,
    type_name: str = "",
):
    field = message.field.add()
    field.name = name
    field.number = number
    field.type = field_type
    field.label = (
        field.LABEL_REPEATED if repeated else field.LABEL_OPTIONAL
    )
    if type_name:
        field.type_name = type_name


def _message_class(message_factory, descriptor):
    getter = getattr(message_factory, "GetMessageClass", None)
    if getter is not None:
        return getter(descriptor)
    return message_factory.MessageFactory().GetPrototype(descriptor)


def _request_to_proto(request: Request):
    runtime = require_grpc()
    value = runtime["request_class"]()
    value.action = request.action
    value.worker_id = request.worker_id
    value.clock = request.clock
    value.keys.extend(request.keys)
    value.mode = request.mode
    for entry in request.entries:
        item = value.entries.add()
        item.key = entry.key
        item.values.extend(entry.values)
        item.op = entry.op
    return value


def _request_from_proto(value) -> Request:
    return Request(
        action=value.action,
        worker_id=value.worker_id,
        clock=value.clock,
        keys=tuple(value.keys),
        entries=tuple(
            ParameterEntry(item.key, tuple(item.values), item.op or "add")
            for item in value.entries
        ),
        mode=value.mode,
    )


def _response_to_proto(response: Response):
    runtime = require_grpc()
    value = runtime["response_class"]()
    value.ok = response.ok
    value.error = response.error
    value.version = response.version
    value.metadata_json = json.dumps(response.metadata, sort_keys=True)
    for entry in response.values:
        item = value.values.add()
        item.key = entry.key
        item.values.extend(entry.values)
        item.op = entry.op
    return value


def _response_from_proto(value) -> Response:
    metadata = json.loads(value.metadata_json) if value.metadata_json else {}
    return Response(
        ok=bool(value.ok),
        error=str(value.error),
        version=int(value.version),
        values=tuple(
            ParameterEntry(item.key, tuple(item.values), item.op or "set")
            for item in value.values
        ),
        metadata=dict(metadata),
    )


class GRPCTransportClient(TransportClient):
    """gRPC unary-unary client using generated-at-runtime protobuf classes."""

    def __init__(self, endpoint: str, *, timeout: float = 30.0):
        super().__init__()
        runtime = require_grpc()
        self.endpoint = str(endpoint)
        self.timeout = float(timeout)
        self._channel = runtime["grpc"].insecure_channel(self.endpoint)
        self._method = self._channel.unary_unary(
            "/kernelleaf.distributed.ParameterServer/Request",
            request_serializer=lambda payload: payload,
            response_deserializer=lambda payload: payload,
        )
        self._response_class = runtime["response_class"]

    def request(self, request: Request) -> Response:
        payload = _request_to_proto(request).SerializeToString()
        response_payload = self._method(payload, timeout=self.timeout)
        self._record_exchange(len(payload), len(response_payload))
        return _response_from_proto(self._response_class.FromString(response_payload))

    def close(self):
        self._channel.close()


def run_grpc_server(
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
    """Process entry point for one gRPC parameter-server shard."""
    set_current_process_affinity(cpu_affinity)
    state = ParameterServerState(
        shard_index,
        num_shards,
        mode=mode,
        barrier_timeout=barrier_timeout,
    )
    server, _ = _build_server(host, port, state)
    server.start()
    ready_event.set()
    server.wait_for_termination()


def _build_server(host: str, port: int, state: ParameterServerState):
    runtime = require_grpc()
    grpc = runtime["grpc"]

    def request_handler(request, context):
        del context
        return _response_to_proto(
            dispatch_request(state, _request_from_proto(request))
        )

    handler = grpc.method_handlers_generic_handler(
        "kernelleaf.distributed.ParameterServer",
        {
            "Request": grpc.unary_unary_rpc_method_handler(
                request_handler,
                request_deserializer=runtime["request_class"].FromString,
                response_serializer=lambda message: message.SerializeToString(),
            )
        },
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=32),
        options=[
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ],
    )
    server.add_generic_rpc_handlers((handler,))
    bound = server.add_insecure_port(f"{host}:{int(port)}")
    if bound != int(port):
        raise RuntimeError(f"gRPC failed to bind {host}:{port}, got {bound}")
    return server, grpc


class GRPCServerHandle:
    """In-process gRPC server useful for tests and embedding."""

    def __init__(self, host: str, port: int, state: ParameterServerState):
        self._server, _ = _build_server(host, int(port), state)
        self._host = str(host)
        self._port = int(port)
        self._thread: Optional[threading.Thread] = None

    @property
    def endpoint(self) -> str:
        return f"{self._host}:{self._port}"

    def start(self):
        if self._thread is not None:
            return self
        self._server.start()
        self._thread = threading.Thread(
            target=self._server.wait_for_termination,
            name=f"kernelleaf-grpc-{self._port}",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self):
        if self._thread is None:
            return
        self._server.stop(grace=0.0).wait(timeout=5.0)
        self._thread.join(timeout=5.0)
        self._thread = None
