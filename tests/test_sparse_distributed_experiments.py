import threading
import time

import numpy as np
import pytest

from experiments.distributed.runtime import (
    ParameterEntry,
    ParameterServerClient,
    ParameterServerState,
    Request,
    Response,
    SocketServerHandle,
    shard_for_key,
    collect_module_parameters,
    pull_module_parameters,
    push_module_parameters,
)


def test_protocol_json_roundtrip():
    request = Request(
        "push",
        worker_id="worker-7",
        clock=12,
        keys=("a", "b"),
        entries=(ParameterEntry("a", (1.5, -2.0)),),
        mode="sync",
    )
    decoded = Request.from_json(request.to_json())
    assert decoded == request

    response = Response(
        version=9,
        values=(ParameterEntry("a", (1.5, -2.0), "set"),),
        metadata={"mode": "async"},
    )
    assert Response.from_json(response.to_json()) == response


def test_sync_barrier_aggregates_all_workers():
    state = ParameterServerState(mode="sync", barrier_timeout=5.0)
    state.register_worker("a")
    state.register_worker("b")
    results = {}
    errors = []

    def push(worker, value):
        try:
            results[worker] = state.push(
                worker, 0, (ParameterEntry("x", (value,), "add"),)
            )
        except Exception as error:
            errors.append(error)

    first = threading.Thread(target=push, args=("a", 1.0))
    second = threading.Thread(target=push, args=("b", 2.0))
    first.start()
    time.sleep(0.05)
    second.start()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not errors
    assert not first.is_alive() and not second.is_alive()
    values, _ = state.pull(("x",))
    assert values[0].values == (3.0,)
    assert results["a"].get("workers", 2) == 2
    assert results["b"].get("workers", 2) == 2


def test_switch_to_async_flushes_pending_barrier():
    state = ParameterServerState(mode="sync", barrier_timeout=5.0)
    state.register_worker("a")
    state.register_worker("b")
    result = {}

    def push():
        result.update(
            state.push("a", 0, (ParameterEntry("x", (4.0,), "add"),))
        )

    thread = threading.Thread(target=push)
    thread.start()
    time.sleep(0.05)
    state.set_mode("async")
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    values, _ = state.pull(("x",))
    assert values[0].values == (4.0,)
    assert result["status"] == "switched"


def test_switch_async_to_sync_keeps_inflight_async_pushes_nonblocking():
    state = ParameterServerState(mode="async", barrier_timeout=5.0)
    state.register_worker("a")
    state.register_worker("b")
    state.set_mode("sync", "a")

    state.push("b", 10, (ParameterEntry("inflight", (3.0,), "add"),))
    values, _ = state.pull(("inflight",))
    assert values[0].values == (3.0,)

    state.set_mode("sync", "b")
    result = {}

    def push_a():
        result.update(
            state.push("a", 1, (ParameterEntry("x", (1.0,), "add"),))
        )

    thread = threading.Thread(target=push_a)
    thread.start()
    time.sleep(0.05)
    state.push("b", 1, (ParameterEntry("x", (2.0,), "add"),))
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    values, _ = state.pull(("x",))
    assert values[0].values == (3.0,)
    assert result["status"] == "applied"


def test_socket_transport_end_to_end():
    state = ParameterServerState(mode="sync", barrier_timeout=5.0)
    server = SocketServerHandle("127.0.0.1", 0, state).start()
    first = ParameterServerClient([server.endpoint], worker_id="a", timeout=5.0)
    second = ParameterServerClient([server.endpoint], worker_id="b", timeout=5.0)
    try:
        first.register()
        second.register()
        errors = []

        def push(client, value):
            try:
                client.push({"x": value}, clock=0)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=push, args=(first, 1.0))
        thread.start()
        time.sleep(0.05)
        push(second, 2.0)
        thread.join(timeout=5.0)
        assert not errors
        assert not thread.is_alive()
        assert first.pull_scalars(("x",))["x"] == 3.0
    finally:
        first.close()
        second.close()
        server.close()


def test_transport_metrics_count_serialized_message_bytes():
    state = ParameterServerState(mode="async", barrier_timeout=5.0)
    server = SocketServerHandle("127.0.0.1", 0, state).start()
    client = ParameterServerClient([server.endpoint], worker_id="metrics", timeout=5.0)
    try:
        client.register()
        client.push({"x": 1.0}, clock=0)
        client.pull_scalars(("x",))
        metrics = client.transport_metrics()
        assert metrics["requests"] == 3
        assert metrics["bytes_sent"] > 0
        assert metrics["bytes_received"] > 0
        assert metrics["message_bytes"] == (
            metrics["bytes_sent"] + metrics["bytes_received"]
        )
        client.reset_transport_metrics()
        assert client.transport_metrics()["message_bytes"] == 0
    finally:
        client.close()
        server.close()


def test_shard_assignment_is_stable():
    assert shard_for_key("feature:42", 4) == shard_for_key("feature:42", 4)
    assert 0 <= shard_for_key("bias", 4) < 4


def test_module_parameter_bridge_roundtrip():
    import kernelleaf as kl

    module = kl.nn.Linear(3, 2, bias=True)
    state = ParameterServerState(mode="async", barrier_timeout=5.0)
    server = SocketServerHandle("127.0.0.1", 0, state).start()
    client = ParameterServerClient([server.endpoint], worker_id="module", timeout=5.0)
    try:
        client.register()
        push_module_parameters(module, client)
        expected = {
            key: tuple(values)
            for key, values in collect_module_parameters(module).items()
        }
        for parameter in module.parameters():
            parameter.cached_data = np.zeros_like(parameter.realize_cached_data())
        pull_module_parameters(module, client)
        actual = collect_module_parameters(module)
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key])
    finally:
        client.close()
        server.close()


def test_grpc_optional_dependency_error_is_actionable():
    from experiments.distributed.runtime.grpc_transport import require_grpc

    try:
        require_grpc()
    except RuntimeError as error:
        assert "grpc" in str(error).lower()
    else:
        pytest.skip("grpcio is installed; optional error path is not exercised")


def test_grpc_transport_when_optional_dependency_is_available():
    pytest.importorskip("grpc")
    import socket

    from experiments.distributed.runtime.grpc_transport import GRPCServerHandle

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    state = ParameterServerState(mode="async", barrier_timeout=5.0)
    server = GRPCServerHandle("127.0.0.1", port, state).start()
    client = ParameterServerClient(
        [server.endpoint], worker_id="grpc-worker", transport="grpc", timeout=5.0
    )
    try:
        client.register()
        client.push({"grpc-key": 2.5}, clock=0)
        assert client.pull_scalars(("grpc-key",))["grpc-key"] == 2.5
    finally:
        client.close()
        server.close()


def test_sparse_lr_baseline_and_epoch_summary_metrics():
    from experiments.distributed.run_experiments import summarize_worker_epochs
    from experiments.distributed.sparse_lr import SparseLRConfig, run_sparse_lr_baseline

    result = run_sparse_lr_baseline(
        config=SparseLRConfig(
            samples=128,
            features=128,
            nnz=4,
            validation_samples=32,
            epochs=2,
            batch_size=16,
            seed=3,
        )
    )
    summary = summarize_worker_epochs(result["worker_results"])
    assert summary["one_epoch_total_time"] > 0
    assert summary["compute_time"] > 0
    assert summary["communication_time"] == 0
    assert summary["message_bytes"] == 0
    assert 0.0 <= result["accuracy"] <= 1.0


def test_sparse_lr_sync_runner_waits_for_all_worker_registrations():
    """Exercise the real spawn path which exposed PR #2's clock-0 race."""
    from experiments.distributed.run_experiments import run_sparse_lr_experiment
    from experiments.distributed.sparse_lr import SparseLRConfig

    result = run_sparse_lr_experiment(
        transport="socket",
        mode="sync",
        workers=2,
        shards=1,
        config=SparseLRConfig(
            samples=64,
            features=64,
            nnz=4,
            validation_samples=16,
            epochs=2,
            batch_size=16,
            seed=7,
        ),
        timeout=10.0,
    )

    assert len(result["worker_results"]) == 2
    assert set(result["server_metadata"][0]["workers"]) == {"lr-0", "lr-1"}
    assert 0.0 <= result["accuracy"] <= 1.0


def test_epoch_summary_uses_max_wall_time_and_sum_worker_work():
    from experiments.distributed.run_experiments import summarize_worker_epochs

    workers = [
        {
            "worker": 0,
            "epoch_metrics": [
                {
                    "wall_seconds": 1.0,
                    "compute_seconds": 0.4,
                    "communication_seconds": 0.2,
                    "message_bytes": 10,
                },
                {
                    "wall_seconds": 2.0,
                    "compute_seconds": 0.8,
                    "communication_seconds": 0.4,
                    "message_bytes": 20,
                },
            ],
        },
        {
            "worker": 1,
            "epoch_metrics": [
                {
                    "wall_seconds": 1.5,
                    "compute_seconds": 0.3,
                    "communication_seconds": 0.1,
                    "message_bytes": 5,
                },
                {
                    "wall_seconds": 2.5,
                    "compute_seconds": 0.6,
                    "communication_seconds": 0.2,
                    "message_bytes": 8,
                },
            ],
        },
    ]
    summary = summarize_worker_epochs(workers)
    assert summary["one_epoch_total_time"] == pytest.approx(2.0)
    assert summary["compute_time"] == pytest.approx(1.05)
    assert summary["communication_time"] == pytest.approx(0.45)
    assert summary["message_bytes"] == pytest.approx(21.5)


def test_physical_core_groups_are_available_and_disjoint():
    pytest.importorskip("psutil")
    from experiments.distributed.runtime.cpu_affinity import (
        available_logical_cpus,
        physical_core_groups,
    )

    available = set(available_logical_cpus())
    groups = physical_core_groups()
    assert groups
    assert all(group for group in groups)
    assert all(set(group) <= available for group in groups)
    flattened = [cpu for group in groups for cpu in group]
    assert len(flattened) == len(set(flattened))
