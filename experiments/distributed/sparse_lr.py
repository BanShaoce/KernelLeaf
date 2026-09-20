"""Distributed sparse logistic regression on the parameter server."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .runtime import ParameterServerClient
from .runtime.cpu_affinity import set_current_process_affinity
from .data import make_sparse_classification


def feature_key(index: int) -> str:
    return f"feature:{int(index)}"


BIAS_KEY = "bias"


@dataclass(frozen=True)
class SparseLRConfig:
    samples: int = 2048
    features: int = 4096
    nnz: int = 8
    validation_samples: int = 512
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 0.8
    seed: int = 17
    worker_delay: float = 0.0
    mode: str = "sync"
    switch_at: int = -1
    switch_to: str = ""


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _scaled_updates(
    gradients: Mapping[str, float],
    *,
    learning_rate: float,
    samples: int,
) -> Dict[str, float]:
    scale = -float(learning_rate) / max(1, int(samples))
    return {key: scale * value for key, value in gradients.items()}


def _train_one_round_sync(
    client: ParameterServerClient,
    data,
    sample_indices: Sequence[int],
    *,
    total_samples: int,
    batch_size: int,
    learning_rate: float,
    clock: int,
) -> Tuple[int, Dict[str, float]]:
    accumulated: Dict[str, float] = {}
    batches = max(1, int(math.ceil(len(sample_indices) / float(batch_size))))
    communication_seconds = 0.0
    compute_seconds = 0.0
    for start in range(0, len(sample_indices), batch_size):
        batch = sample_indices[start : start + batch_size]
        compute_started = time.perf_counter()
        keys = {BIAS_KEY}
        for row in batch:
            keys.update(feature_key(index) for index in data.indices[row])
        compute_seconds += time.perf_counter() - compute_started
        communication_started = time.perf_counter()
        weights = client.pull_scalars(keys)
        communication_seconds += time.perf_counter() - communication_started
        gradients: Dict[str, float] = {}
        compute_started = time.perf_counter()
        for row in batch:
            logit = weights.get(BIAS_KEY, 0.0)
            for index, value in zip(data.indices[row], data.values[row]):
                logit += weights.get(feature_key(int(index)), 0.0) * float(value)
            error = _sigmoid(logit) - float(data.labels[row])
            gradients[BIAS_KEY] = gradients.get(BIAS_KEY, 0.0) + error
            for index, value in zip(data.indices[row], data.values[row]):
                key = feature_key(int(index))
                gradients[key] = gradients.get(key, 0.0) + error * float(value)
        for key, value in gradients.items():
            accumulated[key] = accumulated.get(key, 0.0) + value
        compute_seconds += time.perf_counter() - compute_started
    communication_started = time.perf_counter()
    client.push(
        _scaled_updates(
            accumulated,
            learning_rate=learning_rate,
            samples=total_samples,
        ),
        clock=clock,
        op="add",
    )
    communication_seconds += time.perf_counter() - communication_started
    return batches, {
        "compute_seconds": compute_seconds,
        "communication_seconds": communication_seconds,
    }


def _train_one_round_async(
    client: ParameterServerClient,
    data,
    sample_indices: Sequence[int],
    *,
    batch_size: int,
    learning_rate: float,
    clock_base: int,
    worker_delay: float,
) -> Tuple[int, Dict[str, float]]:
    batches = 0
    communication_seconds = 0.0
    compute_seconds = 0.0
    for start in range(0, len(sample_indices), batch_size):
        batch = sample_indices[start : start + batch_size]
        compute_started = time.perf_counter()
        keys = {BIAS_KEY}
        for row in batch:
            keys.update(feature_key(index) for index in data.indices[row])
        compute_seconds += time.perf_counter() - compute_started
        communication_started = time.perf_counter()
        weights = client.pull_scalars(keys)
        communication_seconds += time.perf_counter() - communication_started
        gradients: Dict[str, float] = {}
        compute_started = time.perf_counter()
        for row in batch:
            logit = weights.get(BIAS_KEY, 0.0)
            for index, value in zip(data.indices[row], data.values[row]):
                logit += weights.get(feature_key(int(index)), 0.0) * float(value)
            error = _sigmoid(logit) - float(data.labels[row])
            gradients[BIAS_KEY] = gradients.get(BIAS_KEY, 0.0) + error
            for index, value in zip(data.indices[row], data.values[row]):
                key = feature_key(int(index))
                gradients[key] = gradients.get(key, 0.0) + error * float(value)
        compute_seconds += time.perf_counter() - compute_started
        updates = _scaled_updates(
            gradients,
            learning_rate=learning_rate,
            samples=len(batch),
        )
        communication_started = time.perf_counter()
        client.push(updates, clock=clock_base + batches, op="add")
        communication_seconds += time.perf_counter() - communication_started
        batches += 1
        if worker_delay:
            time.sleep(worker_delay)
    return batches, {
        "compute_seconds": compute_seconds,
        "communication_seconds": communication_seconds,
    }


def run_sparse_lr_worker(
    endpoints: Sequence[str],
    worker_id: str,
    *,
    workers: int,
    transport: str,
    config: SparseLRConfig,
    timeout: float,
    result_queue,
    cpu_affinity: Optional[Sequence[int]] = None,
    registration_barrier=None,
) -> None:
    """Top-level process entry point for one sparse-LR worker."""
    try:
        set_current_process_affinity(cpu_affinity)
        data = make_sparse_classification(
            samples=config.samples,
            features=config.features,
            nnz=config.nnz,
            seed=config.seed,
        )
        local_rows = np.arange(int(worker_id), config.samples, workers, dtype=np.int64)
        started = time.perf_counter()
        batches = 0
        epoch_metrics = []
        current_mode = config.mode
        client = ParameterServerClient(
            endpoints,
            worker_id=f"lr-{worker_id}",
            transport=transport,
            timeout=timeout,
        )
        client.register()
        # A synchronous server must know the complete worker set before the
        # first clock is submitted.  Otherwise an early worker can complete
        # clock 0 alone while a later registration makes that clock wait for a
        # contribution which can no longer arrive.
        if registration_barrier is not None:
            registration_barrier.wait(timeout=timeout)
        for epoch in range(config.epochs):
            epoch_started = time.perf_counter()
            transport_before = client.transport_metrics()
            switch_communication = 0.0
            if config.switch_at >= 0 and epoch == config.switch_at:
                current_mode = config.switch_to or (
                    "async" if config.mode == "sync" else "sync"
                )
                switch_started = time.perf_counter()
                client.set_mode(current_mode)
                switch_communication = time.perf_counter() - switch_started
            if current_mode == "sync":
                round_batches, round_metrics = _train_one_round_sync(
                    client,
                    data,
                    local_rows,
                    total_samples=config.samples,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    clock=epoch,
                )
            else:
                round_batches, round_metrics = _train_one_round_async(
                    client,
                    data,
                    local_rows,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    clock_base=epoch * max(
                        1, int(math.ceil(len(local_rows) / float(config.batch_size)))
                    ),
                    worker_delay=config.worker_delay,
                )
            batches += round_batches
            epoch_finished = time.perf_counter()
            transport_after = client.transport_metrics()
            message_bytes = sum(
                int(transport_after[key]) - int(transport_before[key])
                for key in ("bytes_sent", "bytes_received")
            )
            epoch_metrics.append(
                {
                    "epoch": epoch,
                    "wall_seconds": epoch_finished - epoch_started,
                    "compute_seconds": round_metrics["compute_seconds"],
                    "communication_seconds": (
                        round_metrics["communication_seconds"]
                        + switch_communication
                    ),
                    "message_bytes": message_bytes,
                }
            )
        client.close()
        result_queue.put(
            {
                "worker": worker_id,
                "batches": batches,
                "epoch_metrics": epoch_metrics,
                "elapsed": time.perf_counter() - started,
            }
        )
    except Exception as error:
        import traceback

        result_queue.put(
            {
                "worker": worker_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )


def _evaluate_sparse_lr_weights(data, weights: Mapping[str, float]) -> Dict[str, float]:
    correct = 0
    log_loss = 0.0
    for row in range(len(data.labels)):
        logit = weights.get(BIAS_KEY, 0.0)
        for index, value in zip(data.indices[row], data.values[row]):
            logit += weights.get(feature_key(int(index)), 0.0) * float(value)
        probability = min(max(_sigmoid(logit), 1e-12), 1.0 - 1e-12)
        label = float(data.labels[row])
        log_loss -= label * math.log(probability) + (1.0 - label) * math.log(
            1.0 - probability
        )
        correct += int((probability >= 0.5) == bool(data.labels[row]))
    samples = len(data.labels)
    return {
        "accuracy": correct / float(samples),
        "log_loss": log_loss / float(samples),
    }


def evaluate_sparse_lr(
    client: ParameterServerClient,
    *,
    config: SparseLRConfig,
) -> Dict[str, float]:
    data = make_sparse_classification(
        samples=config.validation_samples,
        features=config.features,
        nnz=config.nnz,
        seed=config.seed + 100_000,
        task_seed=config.seed,
    )
    keys = {BIAS_KEY}
    for row in range(config.validation_samples):
        keys.update(feature_key(index) for index in data.indices[row])
    weights = client.pull_scalars(keys)
    return _evaluate_sparse_lr_weights(data, weights)


def run_sparse_lr_baseline(*, config: SparseLRConfig) -> Dict[str, object]:
    """Run the same synchronous LR algorithm locally without a server."""
    data = make_sparse_classification(
        samples=config.samples,
        features=config.features,
        nnz=config.nnz,
        seed=config.seed,
    )
    weights: Dict[str, float] = {}
    epoch_metrics = []
    batches = 0
    started = time.perf_counter()
    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        compute_started = time.perf_counter()
        accumulated: Dict[str, float] = {}
        for start in range(0, config.samples, config.batch_size):
            batch = np.arange(
                start,
                min(start + config.batch_size, config.samples),
                dtype=np.int64,
            )
            gradients: Dict[str, float] = {}
            for row in batch:
                logit = weights.get(BIAS_KEY, 0.0)
                for index, value in zip(data.indices[row], data.values[row]):
                    logit += weights.get(feature_key(int(index)), 0.0) * float(value)
                error = _sigmoid(logit) - float(data.labels[row])
                gradients[BIAS_KEY] = gradients.get(BIAS_KEY, 0.0) + error
                for index, value in zip(data.indices[row], data.values[row]):
                    key = feature_key(int(index))
                    gradients[key] = gradients.get(key, 0.0) + error * float(value)
            for key, value in gradients.items():
                accumulated[key] = accumulated.get(key, 0.0) + value
            batches += 1
        for key, value in _scaled_updates(
            accumulated,
            learning_rate=config.learning_rate,
            samples=config.samples,
        ).items():
            weights[key] = weights.get(key, 0.0) + value
        compute_seconds = time.perf_counter() - compute_started
        epoch_metrics.append(
            {
                "epoch": epoch,
                "wall_seconds": time.perf_counter() - epoch_started,
                "compute_seconds": compute_seconds,
                "communication_seconds": 0.0,
                "message_bytes": 0,
            }
        )

    validation = make_sparse_classification(
        samples=config.validation_samples,
        features=config.features,
        nnz=config.nnz,
        seed=config.seed + 100_000,
        task_seed=config.seed,
    )
    metrics = _evaluate_sparse_lr_weights(validation, weights)
    return {
        "experiment": "sparse_logistic_regression",
        "transport": "none",
        "requested_mode": "local",
        "workers": 1,
        "shards": 0,
        "elapsed": time.perf_counter() - started,
        "worker_results": [
            {"worker": 0, "batches": batches, "epoch_metrics": epoch_metrics}
        ],
        **metrics,
    }
