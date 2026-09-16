"""Single-machine synchronous Parameter Server training for MNIST."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import socket
import threading
import time

import numpy as np

import kernelleaf as kl
import kernelleaf.nn as nn
from kernelleaf.data.datasets import MNISTDataset
from kernelleaf.distributed import (
    FramedTransport, JsonCodec, JsonlMonitor, Launcher, MessageType,
    ParameterServer, RoleSpec, Worker,
)
from kernelleaf.distributed.transport import receive_frame, send_frame


@dataclass(frozen=True)
class TrainingConfig:
    world_size: int = 2
    epochs: int = 1
    global_batch_size: int = 128
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    seed: int = 1
    device: str = "cpu"
    data_root: str = "data/MNIST/raw"
    synthetic_samples: int = 0
    hidden_sizes: tuple = (512, 256, 128)
    socket_timeout: float = 30.0
    poll_interval: float = 0.01

    def __post_init__(self):
        if self.world_size not in (1, 2, 4):
            raise ValueError("world_size must be 1, 2, or 4")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.global_batch_size < self.world_size:
            raise ValueError("global_batch_size must be at least world_size")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.synthetic_samples < 0:
            raise ValueError("synthetic_samples must be non-negative")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive dimensions")


class DistributedMNISTMLP(nn.Module):
    def __init__(self, hidden_sizes=(512, 256, 128), device=None):
        super().__init__()
        sizes = (784,) + tuple(hidden_sizes) + (10,)
        layers = []
        for index, (input_size, output_size) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(input_size, output_size, device=device))
            if index < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


def _device(config):
    return kl.cuda(0) if config.device == "cuda" else kl.cpu()


def _make_model(config):
    np.random.seed(config.seed)
    return DistributedMNISTMLP(config.hidden_sizes, _device(config))


def _synchronize(device):
    """Synchronize only for measured CUDA sections; CPU remains a no-op."""
    if device.kind == "cuda":
        cp = device.xp
        with cp.cuda.Device(device.index):
            cp.cuda.get_current_stream().synchronize()


def _load_training_arrays(config):
    if config.synthetic_samples:
        rng = np.random.default_rng(config.seed)
        labels = np.arange(config.synthetic_samples, dtype=np.uint8) % 10
        images = rng.normal(0.0, 0.08, (config.synthetic_samples, 784)).astype(
            np.float32
        )
        images[np.arange(config.synthetic_samples), labels] += 2.5
        return images, labels
    root = Path(config.data_root)
    dataset = MNISTDataset(
        root / "train-images-idx3-ubyte.gz",
        root / "train-labels-idx1-ubyte.gz",
    )
    return dataset.images, dataset.labels


def _global_batches(sample_count, config, epoch):
    if sample_count < config.world_size:
        raise ValueError("dataset must contain at least one sample per Worker")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, epoch]))
    indices = rng.permutation(sample_count)
    batches = [
        indices[start:start + config.global_batch_size]
        for start in range(0, sample_count, config.global_batch_size)
    ]
    if len(batches[-1]) < config.world_size:
        raise ValueError(
            "final global batch is smaller than world_size; choose a compatible "
            "global_batch_size or dataset size"
        )
    return batches


def _connect(host, port, timeout, stop_event):
    deadline = time.monotonic() + timeout
    while not stop_event.is_set():
        try:
            return socket.create_connection((host, port), timeout=min(timeout, 1.0))
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not connect to Parameter Server at {host}:{port}")
            time.sleep(0.02)
    raise RuntimeError("distributed job was stopped before connection")


def _server_process(stop_event, config, host, port):
    model = _make_model(config)
    optimizer = kl.optim.SGD(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    worker_ids = [f"worker-{index}" for index in range(config.world_size)]
    server = ParameterServer(
        model, optimizer, worker_ids, heartbeat_timeout=config.socket_timeout,
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(config.world_size)
    listener.settimeout(0.2)
    threads = []
    try:
        while len(threads) < config.world_size and not stop_event.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            transport = FramedTransport(
                connection, JsonCodec(), timeout=config.socket_timeout
            )
            thread = threading.Thread(
                target=server.serve_connection, args=(transport,), daemon=True
            )
            thread.start()
            threads.append(thread)
        if len(threads) != config.world_size:
            raise RuntimeError("Parameter Server stopped before every Worker connected")
        while any(thread.is_alive() for thread in threads):
            if stop_event.wait(0.05):
                break
        for thread in threads:
            thread.join(timeout=1.0)
    finally:
        listener.close()


def _receive_decoded(sock, codec, max_size):
    started = time.perf_counter()
    payload = receive_frame(sock, max_size)
    receive_time = time.perf_counter() - started
    decode_started = time.perf_counter()
    response = codec.decode(payload)
    decode_time = time.perf_counter() - decode_started
    return response, len(payload) + 4, receive_time, decode_time


def _push_and_wait(worker, transport, local_count, poll_interval):
    codec = transport.codec
    request_started = time.perf_counter()
    request = worker.gradients_message(local_count)
    payload = codec.encode(request)
    serialize_time = time.perf_counter() - request_started
    upload_started = time.perf_counter()
    upload_bytes = send_frame(transport.socket, payload, transport.max_frame_size)
    upload_time = time.perf_counter() - upload_started
    wait_time = 0.0
    download_time = 0.0
    download_bytes = 0
    while True:
        response, wire_bytes, receive_time, decode_time = _receive_decoded(
            transport.socket, codec, transport.max_frame_size
        )
        wait_time += receive_time
        if response.message_type is MessageType.PARAMETERS:
            accept_started = time.perf_counter()
            worker.handle_response(request, response)
            download_time = decode_time + (time.perf_counter() - accept_started)
            download_bytes = wire_bytes
            return {
                "gradient_serialize_time": serialize_time,
                "gradient_upload_time": upload_time,
                "parameter_wait_time": wait_time,
                "parameter_download_time": download_time,
                "optimizer_time": float(response.payload.get("optimizer_time", 0.0)),
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
            }
        worker.handle_response(request, response)
        time.sleep(poll_interval)
        pull = worker.pull_message(worker.step + 1)
        pull_payload = codec.encode(pull)
        send_frame(transport.socket, pull_payload, transport.max_frame_size)
        request = pull


def _worker_process(stop_event, config, host, port, worker_index, metric_queue):
    worker_id = f"worker-{worker_index}"
    images, labels = _load_training_arrays(config)
    model = _make_model(config)
    model.train()
    worker = Worker(worker_id, model)
    loss_fn = nn.SoftmaxLoss()
    sock = _connect(host, port, config.socket_timeout, stop_event)
    transport = FramedTransport(sock, JsonCodec(), timeout=config.socket_timeout)
    try:
        worker.register(transport)
        global_step = 0
        for epoch in range(config.epochs):
            for global_indices in _global_batches(len(images), config, epoch):
                if stop_event.is_set():
                    raise RuntimeError("distributed job stopped")
                step_started = time.perf_counter()
                data_started = time.perf_counter()
                local_indices = np.array_split(
                    global_indices, config.world_size
                )[worker_index]
                device = _device(config)
                inputs = kl.Tensor(images[local_indices], device=device)
                targets = kl.Tensor(labels[local_indices], device=device)
                _synchronize(device)
                data_time = time.perf_counter() - data_started

                _synchronize(device)
                forward_started = time.perf_counter()
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                _synchronize(device)
                forward_time = time.perf_counter() - forward_started
                loss_value = float(loss.numpy())
                predictions = kl.ops.argmax(logits, axis=1).numpy()
                accuracy = float(np.mean(predictions == labels[local_indices]))

                _synchronize(device)
                backward_started = time.perf_counter()
                loss.backward()
                _synchronize(device)
                backward_time = time.perf_counter() - backward_started
                communication = _push_and_wait(
                    worker, transport, len(local_indices), config.poll_interval
                )
                total_time = time.perf_counter() - step_started
                global_step += 1
                metric_queue.put({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "data_time": data_time,
                    "forward_time": forward_time,
                    "backward_time": backward_time,
                    **communication,
                    "total_step_time": total_time,
                    "samples_per_second": len(local_indices) / total_time,
                    "loss": loss_value,
                    "accuracy": accuracy,
                    "worker_id": worker_id,
                    "parameter_version": worker.step,
                })
        worker.shutdown(transport)
    finally:
        transport.close()


def _monitor_process(stop_event, output_path, metric_queue, expected_records):
    del stop_event
    JsonlMonitor(output_path).run(metric_queue, expected_records)


def _reserve_port(host):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def run_distributed_training(config, output_path, *, timeout=120.0):
    """Run one local PS job and return its measured summary."""
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig")
    images, _ = _load_training_arrays(config)
    steps_per_epoch = len(_global_batches(len(images), config, 0))
    expected_records = config.epochs * steps_per_epoch * config.world_size
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()
    launcher = Launcher()
    metric_queue = launcher.queue()
    host = "127.0.0.1"
    port = _reserve_port(host)
    specs = [
        RoleSpec("monitor", _monitor_process,
                 (str(output_path), metric_queue, expected_records)),
        RoleSpec("parameter-server", _server_process,
                 (config, host, port)),
    ]
    specs.extend(
        RoleSpec(f"worker-{index}", _worker_process,
                 (config, host, port, index, metric_queue))
        for index in range(config.world_size)
    )
    result = launcher.run(specs, timeout=timeout)
    metric_queue.close()
    records = [json.loads(line) for line in output_path.read_text(
        encoding="utf-8"
    ).splitlines()]
    return {
        "world_size": config.world_size,
        "steps_per_epoch": steps_per_epoch,
        "records": len(records),
        "elapsed": result.elapsed,
        # End-to-end job throughput includes spawn, communication and monitoring.
        "samples_per_second": (len(images) * config.epochs) / result.elapsed,
        "first_loss": float(np.mean([
            item["loss"] for item in records if item["step"] == 1
        ])),
        "last_loss": float(np.mean([
            item["loss"] for item in records
            if item["step"] == config.epochs * steps_per_epoch
        ])),
        "exitcodes": result.exitcodes,
        "config": asdict(config),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="KernelLeaf single-machine synchronous PS MNIST training"
    )
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-root", default="data/MNIST/raw")
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--metrics", default="runs/distributed_mnist/metrics.jsonl")
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        world_size=args.workers, epochs=args.epochs,
        global_batch_size=args.global_batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        seed=args.seed, device=args.device, data_root=args.data_root,
        synthetic_samples=args.synthetic_samples,
    )
    summary = run_distributed_training(config, args.metrics, timeout=args.timeout)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
