import socket
import threading

import numpy as np
import pytest

import kernelleaf as kl
import kernelleaf.nn as nn
from kernelleaf.distributed import (
    FramedTransport,
    JsonCodec,
    Message,
    MessageType,
    ParameterServer,
    Worker,
    WorkerProtocolError,
    parameter_schema,
)


def _model(device=None):
    model = nn.Linear(1, 1, bias=False, device=device)
    model.weight.cached_data[...] = 0.5
    return model


def _clone_model(model, device=None):
    result = _model(device=device)
    result.load_state_dict(model.state_dict())
    return result


def _local_backward(model, inputs, targets):
    x = kl.Tensor(np.asarray(inputs, dtype=np.float32), device=model.weight.device,
                  requires_grad=False)
    y = kl.Tensor(np.asarray(targets, dtype=np.float32), device=model.weight.device,
                  requires_grad=False)
    difference = model(x) - y
    loss = (difference * difference).sum() / x.shape[0]
    loss.backward()
    return float(loss.numpy())


def _direct_register(server, worker):
    request = worker.registration_message()
    response = server.handle_message(request)
    assert worker.handle_response(request, response)


def _tcp_pair():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname(), timeout=1)
    server, _ = listener.accept()
    listener.close()
    client.settimeout(None)
    server.settimeout(None)
    return server, client


def _start_connection(server):
    server_socket, worker_socket = _tcp_pair()
    server_transport = FramedTransport(server_socket, JsonCodec(), timeout=2)
    worker_transport = FramedTransport(worker_socket, JsonCodec(), timeout=2)
    thread = threading.Thread(
        target=server.serve_connection, args=(server_transport,), daemon=True
    )
    thread.start()
    return worker_transport, thread


def _close_connections(transports, threads):
    for transport in transports:
        transport.close()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_parameter_schema_is_stable_and_optimizer_must_own_global_model():
    class ReverseNames(nn.Module):
        def __init__(self):
            super().__init__()
            self.z = nn.Parameter(np.ones(2, dtype=np.float32))
            self.a = nn.Parameter(np.zeros(3, dtype=np.float32))

    model = ReverseNames()
    schema = parameter_schema(model)
    assert [item.name for item in schema] == ["a", "z"]
    assert schema[0].shape == (3,)
    with pytest.raises(ValueError, match="optimizer must own"):
        ParameterServer(model, kl.optim.SGD([model.a]), ["worker-0"])


def test_registration_rejects_unknown_worker_and_schema_mismatch():
    model = _model()
    server = ParameterServer(
        model, kl.optim.SGD(model.parameters()), ["worker-0"]
    )
    rogue = Worker("rogue", _clone_model(model))
    response = server.handle_message(rogue.registration_message())
    assert response.message_type is MessageType.ERROR
    assert response.payload["code"] == "UNKNOWN_WORKER"

    incompatible = Worker("worker-0", nn.Linear(2, 1, bias=False))
    response = server.handle_message(incompatible.registration_message())
    assert response.message_type is MessageType.ERROR
    assert response.payload["code"] == "SCHEMA_MISMATCH"


def test_two_worker_tcp_update_matches_weighted_single_process_reference():
    server_model = _model()

    class CountingSGD(kl.optim.SGD):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.step_calls = 0

        def step(self):
            self.step_calls += 1
            return super().step()

    optimizer = CountingSGD(server_model.parameters(), lr=0.05)
    server = ParameterServer(
        server_model, optimizer, ["worker-0", "worker-1"]
    )
    workers = [
        Worker("worker-0", _clone_model(server_model)),
        Worker("worker-1", _clone_model(server_model)),
    ]
    connections = [_start_connection(server), _start_connection(server)]
    transports = [item[0] for item in connections]
    threads = [item[1] for item in connections]
    try:
        assert workers[0].register(transports[0])
        assert workers[1].register(transports[1])
        _local_backward(workers[0].model, [[1.0], [2.0]], [[2.0], [4.0]])
        _local_backward(workers[1].model, [[3.0]], [[6.0]])

        assert workers[0].push_gradients(transports[0], 2) is False
        assert optimizer.step_calls == 0
        assert workers[1].push_gradients(transports[1], 1) is True
        assert optimizer.step_calls == 1
        assert server.step == workers[1].step == 1
        assert workers[0].pull_parameters(transports[0], minimum_step=1) is True

        reference = _model()
        reference_optimizer = kl.optim.SGD(reference.parameters(), lr=0.05)
        _local_backward(
            reference,
            [[1.0], [2.0], [3.0]],
            [[2.0], [4.0], [6.0]],
        )
        reference_optimizer.step()
        for model in (server_model, workers[0].model, workers[1].model):
            np.testing.assert_allclose(
                model.weight.numpy(), reference.weight.numpy(),
                rtol=1e-6, atol=1e-7,
            )
        assert all(parameter.grad is None for parameter in server_model.parameters())
        assert server.snapshot()["pending_workers"] == []
    finally:
        _close_connections(transports, threads)


def test_duplicate_stale_and_future_gradients_return_structured_errors():
    model = _model()
    server = ParameterServer(
        model, kl.optim.SGD(model.parameters(), lr=0.1),
        ["worker-0", "worker-1"],
    )
    workers = [Worker("worker-0", _clone_model(model)),
               Worker("worker-1", _clone_model(model))]
    for worker in workers:
        _direct_register(server, worker)
        _local_backward(worker.model, [[1.0]], [[2.0]])

    first = workers[0].gradients_message(1)
    assert server.handle_message(first).message_type is MessageType.HEARTBEAT
    duplicate = server.handle_message(workers[0].gradients_message(1))
    assert duplicate.message_type is MessageType.ERROR
    assert duplicate.payload["code"] == "DUPLICATE_GRADIENT"

    completed = server.handle_message(workers[1].gradients_message(1))
    assert completed.message_type is MessageType.PARAMETERS
    stale = server.handle_message(first)
    assert stale.payload["code"] == "STALE_STEP"
    future_request = Message(
        MessageType.PUSH_GRADIENTS, "future", "worker-0", 2, first.payload
    )
    assert server.handle_message(future_request).payload["code"] == "FUTURE_STEP"


@pytest.mark.parametrize("mutation,code", [
    (lambda payload: payload["gradients"][0].update(
        value=np.zeros((2, 1), dtype=np.float32)), "SHAPE_MISMATCH"),
    (lambda payload: payload["gradients"][0].update(
        value=np.zeros((1, 1), dtype=np.float64)), "DTYPE_MISMATCH"),
    (lambda payload: payload.update(local_sample_count=0), "INVALID_PAYLOAD"),
])
def test_gradient_payload_validation(mutation, code):
    model = _model()
    server = ParameterServer(
        model, kl.optim.SGD(model.parameters()), ["worker-0"]
    )
    worker = Worker("worker-0", _clone_model(model))
    _direct_register(server, worker)
    _local_backward(worker.model, [[1.0]], [[2.0]])
    request = worker.gradients_message(1)
    mutation(request.payload)
    response = server.handle_message(request)
    assert response.message_type is MessageType.ERROR
    assert response.payload["code"] == code
    assert server.step == 0


def test_worker_timeout_fails_the_synchronous_step_and_clears_pending():
    now = [0.0]
    model = _model()
    server = ParameterServer(
        model, kl.optim.SGD(model.parameters()),
        ["worker-0", "worker-1"], heartbeat_timeout=5,
        clock=lambda: now[0],
    )
    workers = [Worker("worker-0", _clone_model(model)),
               Worker("worker-1", _clone_model(model))]
    for worker in workers:
        _direct_register(server, worker)
    _local_backward(workers[0].model, [[1.0]], [[2.0]])
    assert server.handle_message(
        workers[0].gradients_message(1)
    ).message_type is MessageType.HEARTBEAT
    assert server.snapshot()["pending_workers"] == ["worker-0"]
    now[0] = 4
    heartbeat = Message(MessageType.HEARTBEAT, "heartbeat", "worker-0", 0, {})
    assert server.handle_message(heartbeat).message_type is MessageType.HEARTBEAT
    now[0] = 6
    assert server.check_timeouts() == ("worker-1",)
    response = server.handle_message(workers[0].pull_message())
    assert response.message_type is MessageType.ERROR
    assert response.payload["code"] == "WORKER_TIMEOUT"
    assert server.snapshot()["failed"] is True
    assert server.snapshot()["pending_workers"] == []


def test_worker_rejects_server_errors_and_missing_local_gradients():
    worker = Worker("worker-0", _model())
    with pytest.raises(WorkerProtocolError, match="no local gradient"):
        worker.gradients_message(1)
    request = worker.registration_message()
    response = Message(
        MessageType.ERROR, request.request_id, "parameter-server", 0,
        {"code": "SCHEMA_MISMATCH", "message": "bad schema"},
    )
    with pytest.raises(WorkerProtocolError, match="bad schema") as error:
        worker.handle_response(request, response)
    assert error.value.code == "SCHEMA_MISMATCH"


@pytest.mark.skipif(not kl.is_cuda_available(), reason="working CUDA/CuPy unavailable")
def test_gpu_parameter_server_uses_explicit_cpu_network_boundary():
    gpu = kl.cuda(0)
    server_model = _model(device=gpu)
    server = ParameterServer(
        server_model, kl.optim.SGD(server_model.parameters(), lr=0.05),
        ["worker-0"],
    )
    worker = Worker("worker-0", _model())
    _direct_register(server, worker)
    assert worker.model.weight.device == kl.cpu()
    _local_backward(worker.model, [[2.0]], [[4.0]])
    request = worker.gradients_message(1)
    assert isinstance(request.payload["gradients"][0]["value"], np.ndarray)
    response = server.handle_message(request)
    assert response.message_type is MessageType.PARAMETERS
    worker.handle_response(request, response)
    assert server_model.weight.device == gpu
    np.testing.assert_allclose(
        server_model.weight.numpy(), worker.model.weight.numpy()
    )
