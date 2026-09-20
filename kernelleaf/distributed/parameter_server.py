"""Synchronous Parameter Server state machine for KernelLeaf V12.2."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..autograd import Tensor
from ..backend import asnumpy, to_device
from .protocol import Message, MessageType
from .transport import PeerDisconnected, TransportError


SERVER_ID = "parameter-server"


class ParameterServerError(ValueError):
    """A request error that can be returned to a Worker."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str

    def to_dict(self):
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class GradientSubmission:
    local_sample_count: int
    gradients: Tuple[np.ndarray, ...]


def stable_named_parameters(model):
    """Return unique trainable parameters in deterministic name order."""
    values = sorted(model.named_parameters(), key=lambda item: item[0])
    names = [name for name, _ in values]
    if not values:
        raise ValueError("distributed model must contain at least one parameter")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("parameter names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("distributed model contains duplicate parameter names")
    return tuple(values)


def parameter_schema(model):
    return tuple(
        ParameterSpec(name, tuple(parameter.shape), np.dtype(parameter.dtype).str)
        for name, parameter in stable_named_parameters(model)
    )


def _schema_payload(schema):
    return [spec.to_dict() for spec in schema]


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ParameterServerError("INVALID_PAYLOAD", f"{name} must be positive")
    return value


class ParameterServer:
    """Own the sole global model/optimizer and aggregate one synchronous step."""

    def __init__(self, model, optimizer, worker_ids: Iterable[str], *,
                 heartbeat_timeout: Optional[float] = None, clock=None):
        self.model = model
        self.optimizer = optimizer
        self._named_parameters = stable_named_parameters(model)
        self.schema = tuple(
            ParameterSpec(
                name, tuple(parameter.shape), np.dtype(parameter.dtype).str
            )
            for name, parameter in self._named_parameters
        )
        if any(np.dtype(spec.dtype).kind != "f" for spec in self.schema):
            raise ValueError("Parameter Server supports floating parameters only")
        model_ids = {id(parameter) for _, parameter in self._named_parameters}
        optimizer_ids = {id(parameter) for parameter in optimizer.params}
        if model_ids != optimizer_ids or len(optimizer.params) != len(model_ids):
            raise ValueError(
                "optimizer must own every model parameter exactly once"
            )
        workers = tuple(worker_ids)
        if not workers or any(
            not isinstance(worker, str) or not worker for worker in workers
        ):
            raise ValueError("worker_ids must contain non-empty strings")
        if len(set(workers)) != len(workers):
            raise ValueError("worker_ids must be unique")
        if heartbeat_timeout is not None and heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be positive or None")
        self.worker_ids = tuple(sorted(workers))
        self.heartbeat_timeout = heartbeat_timeout
        self._clock = clock or time.monotonic
        self._registered = set()
        self._last_seen: Dict[str, float] = {}
        self._pending: Dict[str, GradientSubmission] = {}
        self._failed_reason = None
        self.last_optimizer_time = 0.0
        self._lock = threading.RLock()
        self.step = 0

    def _response(self, request, message_type, payload):
        return Message(
            message_type=message_type,
            request_id=request.request_id,
            worker_id=SERVER_ID,
            step=self.step,
            payload=payload,
        )

    def _error(self, request, error):
        return self._response(request, MessageType.ERROR, {
            "code": error.code,
            "message": str(error),
            "parameter_version": self.step,
        })

    def _parameter_entries(self):
        return [
            {
                "name": name,
                # This is an explicit device-to-host network boundary.
                "value": asnumpy(parameter.realize_cached_data()).copy(),
            }
            for name, parameter in self._named_parameters
        ]

    def _parameters_response(self, request):
        return self._response(request, MessageType.PARAMETERS, {
            "parameter_version": self.step,
            "parameters": self._parameter_entries(),
            "optimizer_time": self.last_optimizer_time,
        })

    def _waiting_response(self, request):
        return self._response(request, MessageType.HEARTBEAT, {
            "status": "waiting",
            "parameter_version": self.step,
            "received_workers": sorted(self._pending),
            "expected_workers": list(self.worker_ids),
        })

    def _validate_worker(self, worker_id, *, require_registered=True):
        if worker_id not in self.worker_ids:
            raise ParameterServerError(
                "UNKNOWN_WORKER", f"unknown worker {worker_id!r}"
            )
        if require_registered and worker_id not in self._registered:
            raise ParameterServerError(
                "NOT_REGISTERED", f"worker {worker_id!r} is not registered"
            )

    def _validate_schema(self, value):
        if not isinstance(value, list):
            raise ParameterServerError(
                "SCHEMA_MISMATCH", "parameter_schema must be a list"
            )
        if value != _schema_payload(self.schema):
            raise ParameterServerError(
                "SCHEMA_MISMATCH", "worker parameter schema does not match server"
            )

    def _decode_gradients(self, payload):
        if not isinstance(payload, dict):
            raise ParameterServerError("INVALID_PAYLOAD", "payload must be an object")
        count = _validate_positive_integer(
            payload.get("local_sample_count"), "local_sample_count"
        )
        entries = payload.get("gradients")
        if not isinstance(entries, list):
            raise ParameterServerError("INVALID_PAYLOAD", "gradients must be a list")
        expected_names = [spec.name for spec in self.schema]
        actual_names = [
            entry.get("name") if isinstance(entry, dict) else None
            for entry in entries
        ]
        if actual_names != expected_names:
            raise ParameterServerError(
                "PARAMETER_ORDER_MISMATCH",
                f"gradient names/order must be {expected_names}, got {actual_names}",
            )
        gradients = []
        for entry, spec in zip(entries, self.schema):
            if set(entry) != {"name", "value"}:
                raise ParameterServerError(
                    "INVALID_PAYLOAD", f"invalid gradient entry for {spec.name}"
                )
            try:
                # A direct API caller may pass CuPy. Normalize explicitly at
                # the same communication boundary used by JsonCodec.
                gradient = np.asarray(asnumpy(entry["value"]))
            except Exception as error:
                raise ParameterServerError(
                    "INVALID_PAYLOAD", f"invalid gradient for {spec.name}: {error}"
                ) from error
            if tuple(gradient.shape) != spec.shape:
                raise ParameterServerError(
                    "SHAPE_MISMATCH",
                    f"gradient {spec.name} expected {spec.shape}, got "
                    f"{tuple(gradient.shape)}",
                )
            if gradient.dtype.str != spec.dtype:
                raise ParameterServerError(
                    "DTYPE_MISMATCH",
                    f"gradient {spec.name} expected {spec.dtype}, got "
                    f"{gradient.dtype.str}",
                )
            if not np.all(np.isfinite(gradient)):
                raise ParameterServerError(
                    "NONFINITE_GRADIENT", f"gradient {spec.name} is non-finite"
                )
            gradients.append(np.ascontiguousarray(gradient).copy())
        return GradientSubmission(count, tuple(gradients))

    def _apply_pending_gradients(self):
        total_samples = sum(
            submission.local_sample_count
            for submission in self._pending.values()
        )
        for index, (_, parameter) in enumerate(self._named_parameters):
            dtype = np.dtype(self.schema[index].dtype)
            accumulation_dtype = np.float64 if dtype.itemsize >= 8 else np.float32
            average = np.zeros(parameter.shape, dtype=accumulation_dtype)
            for worker_id in self.worker_ids:
                submission = self._pending[worker_id]
                average += submission.gradients[index].astype(
                    accumulation_dtype, copy=False
                ) * (submission.local_sample_count / total_samples)
            average = average.astype(dtype, copy=False)
            parameter.grad = Tensor(
                to_device(average, parameter.device, dtype=parameter.dtype),
                device=parameter.device,
                dtype=parameter.dtype,
                requires_grad=False,
            )
        self._synchronize_parameter_devices()
        optimizer_started = self._clock()
        self.optimizer.step()
        self._synchronize_parameter_devices()
        self.last_optimizer_time = self._clock() - optimizer_started
        self.optimizer.reset_grad()
        self.step += 1
        self._pending.clear()

    def _synchronize_parameter_devices(self):
        """Make optimizer timing include queued CUDA work without eager imports."""
        synchronized = set()
        for _, parameter in self._named_parameters:
            device = parameter.device
            if device.kind != "cuda" or device in synchronized:
                continue
            cp = device.xp
            with cp.cuda.Device(device.index):
                cp.cuda.get_current_stream().synchronize()
            synchronized.add(device)

    def _check_timeouts_locked(self, now=None):
        if self.heartbeat_timeout is None or self._failed_reason is not None:
            return ()
        current = self._clock() if now is None else now
        timed_out = tuple(sorted(
            worker for worker in self._registered
            if current - self._last_seen[worker] > self.heartbeat_timeout
        ))
        if timed_out:
            self._failed_reason = (
                "workers timed out while participating in synchronous training: "
                + ", ".join(timed_out)
            )
            self._pending.clear()
        return timed_out

    def check_timeouts(self, now=None):
        """Mark the synchronous job failed if any registered Worker expires."""
        with self._lock:
            return self._check_timeouts_locked(now)

    def snapshot(self):
        with self._lock:
            return {
                "step": self.step,
                "workers": list(self.worker_ids),
                "registered_workers": sorted(self._registered),
                "pending_workers": sorted(self._pending),
                "failed": self._failed_reason is not None,
                "error": self._failed_reason,
            }

    def handle_message(self, request: Message):
        """Apply one request atomically and always return a protocol Message."""
        with self._lock:
            try:
                self._check_timeouts_locked()
                if self._failed_reason is not None:
                    raise ParameterServerError(
                        "WORKER_TIMEOUT", self._failed_reason
                    )
                self._validate_worker(
                    request.worker_id,
                    require_registered=request.message_type is not MessageType.REGISTER,
                )
                if request.message_type is MessageType.REGISTER:
                    if not isinstance(request.payload, dict):
                        raise ParameterServerError(
                            "INVALID_PAYLOAD", "REGISTER payload must be an object"
                        )
                    self._validate_schema(request.payload.get("parameter_schema"))
                    self._registered.add(request.worker_id)
                    self._last_seen[request.worker_id] = self._clock()
                    return self._parameters_response(request)

                self._last_seen[request.worker_id] = self._clock()
                if request.message_type is MessageType.HEARTBEAT:
                    return self._response(request, MessageType.HEARTBEAT, {
                        "status": "ok", "parameter_version": self.step,
                    })
                if request.message_type is MessageType.PULL_PARAMETERS:
                    if request.step > self.step + 1:
                        raise ParameterServerError(
                            "FUTURE_STEP",
                            f"requested step {request.step}, server is at {self.step}",
                        )
                    if request.step > self.step:
                        return self._waiting_response(request)
                    return self._parameters_response(request)
                if request.message_type is MessageType.PUSH_GRADIENTS:
                    if request.step < self.step:
                        raise ParameterServerError(
                            "STALE_STEP",
                            f"gradient step {request.step}, server is at {self.step}",
                        )
                    if request.step > self.step:
                        raise ParameterServerError(
                            "FUTURE_STEP",
                            f"gradient step {request.step}, server is at {self.step}",
                        )
                    if request.worker_id in self._pending:
                        raise ParameterServerError(
                            "DUPLICATE_GRADIENT",
                            f"worker {request.worker_id!r} already submitted step "
                            f"{self.step}",
                        )
                    self._pending[request.worker_id] = self._decode_gradients(
                        request.payload
                    )
                    if set(self._pending) != set(self.worker_ids):
                        return self._waiting_response(request)
                    self._apply_pending_gradients()
                    return self._parameters_response(request)
                if request.message_type is MessageType.SHUTDOWN:
                    return self._response(request, MessageType.SHUTDOWN, {
                        "status": "acknowledged",
                        "parameter_version": self.step,
                    })
                raise ParameterServerError(
                    "UNSUPPORTED_MESSAGE",
                    f"server cannot handle {request.message_type.value}",
                )
            except ParameterServerError as error:
                return self._error(request, error)

    def serve_connection(self, transport):
        """Serve one connected Worker until shutdown or disconnect."""
        while True:
            try:
                request = transport.receive()
                response = self.handle_message(request)
                transport.send(response)
                if request.message_type is MessageType.SHUTDOWN:
                    return
            except (PeerDisconnected, TransportError):
                return
