import json
import time

import numpy as np
import pytest

from apps.distributed_mnist import (
    TrainingConfig, _global_batches, run_distributed_training,
)
from kernelleaf.distributed import (
    DistributedProcessError, METRIC_FIELDS, JsonlMonitor, Launcher,
    MetricValidationError, RoleSpec,
)


def _fail_role(stop_event):
    del stop_event
    raise RuntimeError("intentional Worker failure")


def _blocking_role(stop_event):
    while not stop_event.wait(0.01):
        pass


def _metric(**updates):
    record = {
        "epoch": 1, "step": 1, "data_time": 0.1,
        "forward_time": 0.2, "backward_time": 0.3,
        "gradient_serialize_time": 0.01, "gradient_upload_time": 0.02,
        "parameter_wait_time": 0.03, "parameter_download_time": 0.04,
        "optimizer_time": 0.005, "total_step_time": 0.7,
        "samples_per_second": 10.0, "upload_bytes": 100,
        "download_bytes": 200, "loss": 2.0, "accuracy": 0.25,
        "worker_id": "worker-0", "parameter_version": 1,
    }
    record.update(updates)
    return record


def test_launcher_always_uses_spawn_context():
    assert Launcher().context.get_start_method() == "spawn"


def test_launcher_terminates_other_roles_when_one_fails():
    started = time.perf_counter()
    with pytest.raises(DistributedProcessError, match="intentional Worker failure"):
        Launcher(shutdown_timeout=1).run([
            RoleSpec("worker-fails", _fail_role),
            RoleSpec("parameter-server-blocks", _blocking_role),
        ], timeout=10)
    assert time.perf_counter() - started < 8


def test_monitor_validates_and_appends_jsonl(tmp_path):
    output = tmp_path / "nested" / "metrics.jsonl"
    monitor = JsonlMonitor(output)
    monitor.write(_metric(step=1))
    monitor.write(_metric(step=2, parameter_version=2))
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2]
    assert set(METRIC_FIELDS).issubset(records[0])
    with pytest.raises(MetricValidationError, match="missing fields"):
        monitor.write({"worker_id": "worker-0"})
    with pytest.raises(MetricValidationError, match="finite"):
        monitor.write(_metric(loss=float("nan")))


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_strong_scaling_partitions_global_batches_without_overlap(world_size):
    config = TrainingConfig(
        world_size=world_size, global_batch_size=8, synthetic_samples=20,
        hidden_sizes=(4,),
    )
    batches = _global_batches(20, config, epoch=0)
    assert [len(batch) for batch in batches] == [8, 8, 4]
    for batch in batches:
        shards = np.array_split(batch, world_size)
        assert all(len(shard) > 0 for shard in shards)
        assert sorted(np.concatenate(shards).tolist()) == sorted(batch.tolist())
        assert sum(map(len, shards)) == len(set(np.concatenate(shards).tolist()))


def test_two_worker_spawn_smoke_lowers_loss_and_exits(tmp_path):
    output = tmp_path / "metrics.jsonl"
    config = TrainingConfig(
        world_size=2,
        epochs=3,
        global_batch_size=20,
        learning_rate=0.1,
        synthetic_samples=40,
        hidden_sizes=(16,),
        socket_timeout=10.0,
    )
    summary = run_distributed_training(config, output, timeout=60.0)
    assert summary["records"] == 12
    assert summary["last_loss"] < summary["first_loss"]
    assert set(summary["exitcodes"].values()) == {0}
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert {record["worker_id"] for record in records} == {
        "worker-0", "worker-1"
    }
    assert {record["parameter_version"] for record in records} == set(range(1, 7))
    assert all(set(METRIC_FIELDS).issubset(record) for record in records)


def test_invalid_final_batch_is_rejected_before_process_launch(tmp_path):
    config = TrainingConfig(
        world_size=4, global_batch_size=8, synthetic_samples=10,
        hidden_sizes=(4,),
    )
    with pytest.raises(ValueError, match="final global batch"):
        run_distributed_training(config, tmp_path / "metrics.jsonl")
