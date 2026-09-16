"""Worker-side synchronous Parameter Server protocol for V12.2."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..backend import asnumpy, to_device
from .parameter_server import parameter_schema, stable_named_parameters
from .protocol import Message, MessageType


class WorkerProtocolError(RuntimeError):
    """Raised when the server rejects a Worker request or sends invalid state."""

    def __init__(self, message, *, code=None):
        super().__init__(message)
        self.code = code


class Worker:
    """Compute/transport helper that never owns or steps an optimizer."""

    def __init__(self, worker_id, model):
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a non-empty string")
        self.worker_id = worker_id
        self.model = model
        self._named_parameters = stable_named_parameters(model)
        self.schema = parameter_schema(model)
        self.step = 0
        self.registered = False
        self._request_counter = 0

    def _request_id(self):
        self._request_counter += 1
        return f"{self.worker_id}-{self._request_counter}"

    def _message(self, message_type, payload, *, step=None):
        return Message(
            message_type=message_type,
            request_id=self._request_id(),
            worker_id=self.worker_id,
            step=self.step if step is None else step,
            payload=payload,
        )

    def registration_message(self):
        return self._message(MessageType.REGISTER, {
            "parameter_schema": [spec.to_dict() for spec in self.schema]
        })

    def pull_message(self, minimum_step=None):
        requested = self.step if minimum_step is None else minimum_step
        return self._message(MessageType.PULL_PARAMETERS, {}, step=requested)

    def gradients_message(self, local_sample_count):
        if isinstance(local_sample_count, bool) \
                or not isinstance(local_sample_count, int) \
                or local_sample_count <= 0:
            raise ValueError("local_sample_count must be a positive integer")
        entries = []
        for name, parameter in self._named_parameters:
            gradient_tensor = getattr(parameter, "grad", None)
            if gradient_tensor is None:
                raise WorkerProtocolError(
                    f"parameter {name!r} has no local gradient"
                )
            gradient = gradient_tensor.realize_cached_data()
            if gradient_tensor.device != parameter.device:
                raise WorkerProtocolError(
                    f"gradient device mismatch for {name}: parameter is on "
                    f"{parameter.device}, gradient is on {gradient_tensor.device}"
                )
            host_gradient = asnumpy(gradient)
            if host_gradient.dtype != parameter.dtype:
                # KernelLeaf eager scalar arithmetic can promote a local
                # gradient. Normalize explicitly at the wire boundary; the PS
                # still rejects any mismatched dtype received on the network.
                host_gradient = host_gradient.astype(parameter.dtype)
            entries.append({
                "name": name,
                # Explicit GPU-to-CPU copy at the network boundary.
                "value": host_gradient.copy(),
            })
        return self._message(MessageType.PUSH_GRADIENTS, {
            "local_sample_count": local_sample_count,
            "gradients": entries,
        })

    def shutdown_message(self):
        return self._message(MessageType.SHUTDOWN, {})

    def _raise_error(self, response):
        payload = response.payload
        code = payload.get("code") if isinstance(payload, dict) else None
        message = payload.get("message") if isinstance(payload, dict) else None
        raise WorkerProtocolError(
            message or "Parameter Server returned an invalid error", code=code
        )

    def accept_parameters(self, response):
        if response.message_type is MessageType.ERROR:
            self._raise_error(response)
        if response.message_type is not MessageType.PARAMETERS:
            raise WorkerProtocolError(
                f"expected PARAMETERS, got {response.message_type.value}"
            )
        payload = response.payload
        if not isinstance(payload, dict):
            raise WorkerProtocolError("PARAMETERS payload must be an object")
        version = payload.get("parameter_version")
        if isinstance(version, bool) or not isinstance(version, int) \
                or version < self.step:
            raise WorkerProtocolError(
                f"invalid parameter version {version!r}; Worker is at {self.step}"
            )
        entries = payload.get("parameters")
        if not isinstance(entries, list):
            raise WorkerProtocolError("parameters must be a list")
        names = [
            entry.get("name") if isinstance(entry, dict) else None
            for entry in entries
        ]
        expected_names = [spec.name for spec in self.schema]
        if names != expected_names:
            raise WorkerProtocolError(
                f"parameter names/order must be {expected_names}, got {names}"
            )
        by_name: Dict[str, object] = dict(self._named_parameters)
        for entry, spec in zip(entries, self.schema):
            if set(entry) != {"name", "value"}:
                raise WorkerProtocolError(
                    f"invalid parameter entry for {spec.name}"
                )
            value = np.asarray(asnumpy(entry["value"]))
            if tuple(value.shape) != spec.shape:
                raise WorkerProtocolError(
                    f"parameter {spec.name} expected shape {spec.shape}, got "
                    f"{tuple(value.shape)}"
                )
            if value.dtype.str != spec.dtype:
                raise WorkerProtocolError(
                    f"parameter {spec.name} expected dtype {spec.dtype}, got "
                    f"{value.dtype.str}"
                )
            destination = by_name[spec.name]
            destination.cached_data = to_device(
                value, destination.device, dtype=destination.dtype
            )
            destination.grad = None
        self.step = version
        self.registered = True
        return True

    def handle_response(self, request, response):
        if response.request_id != request.request_id:
            raise WorkerProtocolError(
                f"response request_id {response.request_id!r} does not match "
                f"{request.request_id!r}"
            )
        if response.message_type is MessageType.ERROR:
            self._raise_error(response)
        if response.message_type is MessageType.PARAMETERS:
            return self.accept_parameters(response)
        if response.message_type is MessageType.HEARTBEAT:
            return False
        if response.message_type is MessageType.SHUTDOWN:
            return True
        raise WorkerProtocolError(
            f"unexpected response type {response.message_type.value}"
        )

    def exchange(self, transport, request):
        transport.send(request)
        return self.handle_response(request, transport.receive())

    def register(self, transport):
        return self.exchange(transport, self.registration_message())

    def pull_parameters(self, transport, minimum_step=None):
        return self.exchange(transport, self.pull_message(minimum_step))

    def push_gradients(self, transport, local_sample_count):
        if not self.registered:
            raise WorkerProtocolError("Worker must register before pushing gradients")
        return self.exchange(
            transport, self.gradients_message(local_sample_count)
        )

    def shutdown(self, transport):
        return self.exchange(transport, self.shutdown_message())
