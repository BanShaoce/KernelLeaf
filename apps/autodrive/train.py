"""Train the KernelLeaf dual-head AutoDrive ResNet with V11 telemetry."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import threading
import time

import numpy as np

import kernelleaf as kl
import kernelleaf.nn as nn
from kernelleaf.data import DataLoader
from kernelleaf.distributed import (
    FramedTransport, JsonCodec, Launcher, ParameterServer, RoleSpec, Worker,
    connect_with_retry, monitor_process, push_gradients_and_wait,
    reserve_local_port,
)

from .dataset import AutoDriveDataset, DEFAULT_MEAN, DEFAULT_STD
from .model import AutoDriveResNet
from .artifacts import build_inference_config, save_inference_config
from .config import map_slug
from .monitoring import (
    CallbackList, JSONLRunLogger, default_run_id, summarize_records,
)


def mse_loss(prediction, target):
    difference = prediction - target
    return (difference * difference).sum() / prediction.shape[0]


def train_epoch(model, loader, optimizer, lambda_throttle=1.0,
                grad_clip_norm=0.0):
    model.train()
    totals = {"loss": 0.0, "steer_loss": 0.0, "throttle_loss": 0.0}
    samples = 0
    batches = 0
    started = time.perf_counter()
    for images, steering_target, throttle_target in loader:
        optimizer.reset_grad()
        steering, throttle = model(images)
        steering_loss = mse_loss(steering, steering_target)
        throttle_loss = mse_loss(throttle, throttle_target)
        loss = steering_loss + lambda_throttle * throttle_loss
        loss.backward()
        if grad_clip_norm > 0:
            optimizer.clip_grad_norm(grad_clip_norm)
        optimizer.step()
        batch = images.shape[0]
        totals["loss"] += float(loss.numpy()) * batch
        totals["steer_loss"] += float(steering_loss.numpy()) * batch
        totals["throttle_loss"] += float(throttle_loss.numpy()) * batch
        samples += batch
        batches += 1
    if not samples:
        raise ValueError("training loader produced no samples")
    return {
        **{key: value / samples for key, value in totals.items()},
        "samples": samples,
        "batches": batches,
        "epoch_seconds": time.perf_counter() - started,
    }


def evaluate_model(model, loader, lambda_throttle=1.0):
    model.eval()
    totals = {
        "loss": 0.0,
        "steer_loss": 0.0,
        "throttle_loss": 0.0,
        "steer_mae": 0.0,
        "throttle_mae": 0.0,
    }
    samples = 0
    for images, steering_target, throttle_target in loader:
        steering, throttle = model(images)
        steering_loss = mse_loss(steering, steering_target)
        throttle_loss = mse_loss(throttle, throttle_target)
        batch = images.shape[0]
        totals["steer_loss"] += float(steering_loss.numpy()) * batch
        totals["throttle_loss"] += float(throttle_loss.numpy()) * batch
        totals["loss"] += float(
            steering_loss.numpy() + lambda_throttle * throttle_loss.numpy()
        ) * batch
        totals["steer_mae"] += np.abs(
            steering.numpy() - steering_target.numpy()
        ).sum()
        totals["throttle_mae"] += np.abs(
            throttle.numpy() - throttle_target.numpy()
        ).sum()
        samples += batch
    if not samples:
        raise ValueError("validation loader produced no samples")
    return {**{key: float(value) / samples for key, value in totals.items()},
            "samples": samples}


def selected_manifest_map(manifest_path, requested):
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("exactly one map name is required")
    requested = requested.strip()
    available = set()
    with Path(manifest_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                if "map_name" in record:
                    available.add(record["map_name"])
    if requested not in available:
        raise ValueError(
            f"unknown map {requested!r}; available maps: {sorted(available)}"
        )
    return requested


def default_checkpoint_path(map_name):
    return Path("checkpoints") / f"autodrive_{map_slug(map_name)}.npz"


def _configuration(args):
    return {
        "manifest": str(args.manifest),
        "image_size": [args.image_height, args.image_width],
        "base_channels": args.base_channels,
        "blocks": [1, 1, 1],
        "throttle_min": args.throttle_min,
        "throttle_max": args.throttle_max,
        "lambda_throttle": args.lambda_throttle,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "seed": args.seed,
        "map": args.map,
    }


@dataclass(frozen=True)
class AutoDriveBenchmarkConfig:
    """Pickle-safe configuration for the first V16 PS benchmark slice."""

    manifest: str = "data/DonkeyCar/manifest.jsonl"
    map_name: str = "mountain-track"
    world_size: int = 1
    epochs: int = 10
    global_batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_throttle: float = 1.0
    base_channels: int = 16
    image_height: int = 60
    image_width: int = 80
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    seed: int = 7
    device: str = "cpu"
    socket_timeout: float = 120.0
    poll_interval: float = 0.01

    def __post_init__(self):
        if self.world_size not in (1, 2, 4):
            raise ValueError("world_size must be 1, 2, or 4")
        if self.epochs <= 0 or self.global_batch_size < self.world_size:
            raise ValueError(
                "epochs must be positive and global_batch_size >= world_size"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.lambda_throttle < 0 or self.base_channels <= 0:
            raise ValueError("invalid loss/model configuration")
        if min(self.image_height, self.image_width) <= 0:
            raise ValueError("image dimensions must be positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.socket_timeout <= 0 or self.poll_interval <= 0:
            raise ValueError("distributed timeouts must be positive")
        if not isinstance(self.map_name, str) or not self.map_name.strip():
            raise ValueError("map_name must be non-empty")


def _benchmark_device(config):
    return kl.cuda(0) if config.device == "cuda" else kl.cpu()


def _benchmark_model(config):
    np.random.seed(config.seed)
    return AutoDriveResNet(
        base_channels=config.base_channels,
        throttle_min=config.throttle_min,
        throttle_max=config.throttle_max,
        device=_benchmark_device(config),
    )


def _benchmark_dataset(config, split, *, augment):
    return AutoDriveDataset(
        config.manifest, split,
        image_size=(config.image_height, config.image_width),
        augment=augment, seed=config.seed, map_name=config.map_name,
    )


def _benchmark_checkpoint_config(config, mode):
    return {
        "manifest": str(config.manifest),
        "image_size": [config.image_height, config.image_width],
        "base_channels": config.base_channels,
        "blocks": [1, 1, 1],
        "throttle_min": config.throttle_min,
        "throttle_max": config.throttle_max,
        "lambda_throttle": config.lambda_throttle,
        "batch_size": config.global_batch_size,
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip_norm": 0.0,
        "seed": config.seed,
        "map": config.map_name,
        "benchmark_mode": mode,
        "world_size": 0 if mode == "single_process" else config.world_size,
    }


def _benchmark_global_batches(sample_count, config, epoch):
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
            "global_batch_size"
        )
    return batches


def _synchronize_benchmark_device(device):
    if device.kind == "cuda":
        cp = device.xp
        with cp.cuda.Device(device.index):
            cp.cuda.get_current_stream().synchronize()


def _benchmark_forward_loss(model, images, steering_target, throttle_target,
                            lambda_throttle):
    steering, throttle = model(images)
    steering_loss = mse_loss(steering, steering_target)
    throttle_loss = mse_loss(throttle, throttle_target)
    loss = steering_loss + lambda_throttle * throttle_loss
    return loss, steering_loss, throttle_loss


def _evaluate_benchmark_checkpoint(config, checkpoint_path):
    device = _benchmark_device(config)
    model = _benchmark_model(config)
    kl.load_checkpoint(checkpoint_path, model)
    dataset = _benchmark_dataset(config, "val", augment=False)
    loader = DataLoader(
        dataset, batch_size=config.global_batch_size, shuffle=False,
        device=device, num_workers=0,
    )
    return evaluate_model(model, loader, config.lambda_throttle)


def run_autodrive_single_benchmark(config, checkpoint_path):
    """Run the no-distribution reference with the same batches and model."""
    if not isinstance(config, AutoDriveBenchmarkConfig):
        raise TypeError("config must be AutoDriveBenchmarkConfig")
    started = time.perf_counter()
    device = _benchmark_device(config)
    dataset = _benchmark_dataset(config, "train", augment=True)
    model = _benchmark_model(config)
    model.train()
    optimizer = kl.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    first_loss = None
    last_loss = None
    steps = 0
    for epoch in range(config.epochs):
        dataset.set_epoch(epoch)
        for indices in _benchmark_global_batches(len(dataset), config, epoch):
            arrays = dataset.get_batch(indices)
            images, steering_target, throttle_target = (
                kl.Tensor(value, device=device) for value in arrays
            )
            optimizer.reset_grad()
            loss, _, _ = _benchmark_forward_loss(
                model, images, steering_target, throttle_target,
                config.lambda_throttle,
            )
            value = float(loss.numpy())
            loss.backward()
            optimizer.step()
            first_loss = value if first_loss is None else first_loss
            last_loss = value
            steps += 1
    _synchronize_benchmark_device(device)
    kl.save_checkpoint(
        checkpoint_path, model, epoch=config.epochs,
        config=_benchmark_checkpoint_config(config, "single_process"),
        normalization=dataset.normalization,
    )
    elapsed = time.perf_counter() - started
    validation = _evaluate_benchmark_checkpoint(config, checkpoint_path)
    return {
        "mode": "single_process", "workers": 0,
        "elapsed": elapsed,
        "samples_per_second": len(dataset) * config.epochs / elapsed,
        "steps": steps, "first_loss": first_loss, "last_loss": last_loss,
        "communication_ratio": 0.0,
        "validation": validation, "checkpoint": str(Path(checkpoint_path)),
    }


def _autodrive_ps_process(stop_event, config, host, port):
    model = _benchmark_model(config)
    optimizer = kl.optim.Adam(
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
            raise RuntimeError("PS stopped before every AutoDrive Worker connected")
        while any(thread.is_alive() for thread in threads):
            if stop_event.wait(0.05):
                break
        for thread in threads:
            thread.join(timeout=1.0)
    finally:
        listener.close()


def _autodrive_worker_process(stop_event, config, host, port, worker_index,
                              metric_queue, checkpoint_path):
    worker_id = f"worker-{worker_index}"
    device = _benchmark_device(config)
    dataset = _benchmark_dataset(config, "train", augment=True)
    model = _benchmark_model(config)
    model.train()
    worker = Worker(worker_id, model)
    sock = connect_with_retry(
        host, port, config.socket_timeout, stop_event
    )
    transport = FramedTransport(
        sock, JsonCodec(), timeout=config.socket_timeout
    )
    global_step = 0
    try:
        worker.register(transport)
        for epoch in range(config.epochs):
            dataset.set_epoch(epoch)
            for global_indices in _benchmark_global_batches(
                    len(dataset), config, epoch):
                step_started = time.perf_counter()
                data_started = time.perf_counter()
                indices = np.array_split(
                    global_indices, config.world_size
                )[worker_index]
                arrays = dataset.get_batch(indices)
                images, steering_target, throttle_target = (
                    kl.Tensor(value, device=device) for value in arrays
                )
                _synchronize_benchmark_device(device)
                data_time = time.perf_counter() - data_started

                forward_started = time.perf_counter()
                loss, steering_loss, throttle_loss = _benchmark_forward_loss(
                    model, images, steering_target, throttle_target,
                    config.lambda_throttle,
                )
                _synchronize_benchmark_device(device)
                forward_time = time.perf_counter() - forward_started
                values = (
                    float(loss.numpy()), float(steering_loss.numpy()),
                    float(throttle_loss.numpy()),
                )

                backward_started = time.perf_counter()
                loss.backward()
                _synchronize_benchmark_device(device)
                backward_time = time.perf_counter() - backward_started
                communication = push_gradients_and_wait(
                    worker, transport, len(indices), config.poll_interval
                )
                _synchronize_benchmark_device(device)
                total_time = time.perf_counter() - step_started
                global_step += 1
                metric_queue.put({
                    "epoch": epoch + 1, "step": global_step,
                    "data_time": data_time, "forward_time": forward_time,
                    "backward_time": backward_time, **communication,
                    "total_step_time": total_time,
                    "samples_per_second": len(indices) / total_time,
                    "loss": values[0], "accuracy": 0.0,
                    "worker_id": worker_id,
                    "parameter_version": worker.step,
                    "steer_loss": values[1], "throttle_loss": values[2],
                    "local_sample_count": len(indices),
                })
        if worker_index == 0:
            # Rank 0 owns the final synchronized parameters and its local
            # BatchNorm running statistics, matching common non-SyncBN DDP.
            kl.save_checkpoint(
                checkpoint_path, model, epoch=config.epochs,
                config=_benchmark_checkpoint_config(
                    config, f"ps_{config.world_size}_workers"
                ),
                normalization=dataset.normalization,
            )
        worker.shutdown(transport)
    finally:
        transport.close()


def _weighted_step_loss(records, step):
    selected = [record for record in records if record["step"] == step]
    samples = sum(record["local_sample_count"] for record in selected)
    return sum(
        record["loss"] * record["local_sample_count"] for record in selected
    ) / samples


def run_autodrive_ps_benchmark(config, checkpoint_path, metrics_path,
                               *, timeout=7200.0):
    """Run one synchronous PS AutoDrive job and return measured results."""
    if not isinstance(config, AutoDriveBenchmarkConfig):
        raise TypeError("config must be AutoDriveBenchmarkConfig")
    dataset = _benchmark_dataset(config, "train", augment=True)
    steps_per_epoch = len(_benchmark_global_batches(len(dataset), config, 0))
    expected = steps_per_epoch * config.epochs * config.world_size
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        metrics_path.unlink()
    launcher = Launcher()
    metric_queue = launcher.queue()
    host = "127.0.0.1"
    port = reserve_local_port(host)
    specs = [
        RoleSpec("monitor", monitor_process,
                 (str(metrics_path), metric_queue, expected)),
        RoleSpec("parameter-server", _autodrive_ps_process,
                 (config, host, port)),
    ]
    specs.extend(
        RoleSpec(
            f"worker-{index}", _autodrive_worker_process,
            (config, host, port, index, metric_queue, str(checkpoint_path)),
        )
        for index in range(config.world_size)
    )
    result = launcher.run(specs, timeout=timeout)
    metric_queue.close()
    records = [
        json.loads(line) for line in metrics_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    validation = _evaluate_benchmark_checkpoint(config, checkpoint_path)
    final_step = steps_per_epoch * config.epochs
    communication_fields = (
        "gradient_serialize_time", "gradient_upload_time",
        "parameter_wait_time", "parameter_download_time",
    )
    total_step_time = sum(record["total_step_time"] for record in records)
    communication_time = sum(
        sum(record[field] for field in communication_fields)
        for record in records
    )
    return {
        "mode": f"ps_{config.world_size}_workers",
        "workers": config.world_size, "elapsed": result.elapsed,
        "samples_per_second": len(dataset) * config.epochs / result.elapsed,
        "steps": final_step,
        "first_loss": _weighted_step_loss(records, 1),
        "last_loss": _weighted_step_loss(records, final_step),
        "validation": validation,
        "communication_ratio": communication_time / total_step_time,
        "checkpoint": str(Path(checkpoint_path)),
        "metrics": str(metrics_path), "exitcodes": result.exitcodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/DonkeyCar/manifest.jsonl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--grad-clip-norm", type=float, default=0.0,
        help="clip the global gradient norm; 0 disables clipping",
    )
    parser.add_argument("--lambda-throttle", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=60)
    parser.add_argument("--image-width", type=int, default=80)
    parser.add_argument("--throttle-min", type=float, default=0.0)
    parser.add_argument("--throttle-max", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--map", required=True,
        help="the single manifest map name to train",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="output NPZ; defaults to checkpoints/autodrive_<map>.npz",
    )
    parser.add_argument(
        "--run-config", default=None,
        help="JSON inference config; defaults to the checkpoint path with .json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--runs-root", default="runs",
        help="V11 monitoring run directory root",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="monitoring run name; defaults to the map name plus a timestamp",
    )
    parser.add_argument(
        "--no-monitor-log", action="store_true",
        help="disable V11 JSONL logging; training otherwise behaves normally",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.lambda_throttle < 0 or args.grad_clip_norm < 0:
        parser.error(
            "epochs must be positive; lambda-throttle and grad-clip-norm "
            "must be non-negative"
        )
    np.random.seed(args.seed)
    args.map = selected_manifest_map(args.manifest, args.map)
    if args.checkpoint is None:
        args.checkpoint = str(default_checkpoint_path(args.map))
    device = kl.cuda(0) if args.device == "cuda" else kl.cpu()
    image_size = (args.image_height, args.image_width)
    resume_info = kl.inspect_checkpoint(args.checkpoint) if args.resume else None
    normalization = (
        resume_info["normalization"] if resume_info is not None else {}
    )
    mean = normalization.get("mean", DEFAULT_MEAN)
    std = normalization.get("std", DEFAULT_STD)
    train_dataset = AutoDriveDataset(
        args.manifest, "train", image_size=image_size, augment=True,
        seed=args.seed, mean=mean, std=std, map_name=args.map,
    )
    validation_dataset = AutoDriveDataset(
        args.manifest, "val", image_size=image_size, augment=False,
        seed=args.seed, mean=mean, std=std, map_name=args.map,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "device": device,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": device.kind == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, seed=args.seed, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    config = _configuration(args)
    config["throttle_mode"] = (
        "predicted" if train_dataset.has_recorded_throttle else "fixed"
    )
    config["fixed_throttle"] = 0.2
    model = AutoDriveResNet(
        base_channels=args.base_channels,
        throttle_min=args.throttle_min,
        throttle_max=args.throttle_max,
        device=device,
    )
    optimizer = kl.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 1
    if args.resume:
        metadata = kl.load_checkpoint(
            args.checkpoint, model, optimizer=optimizer
        )
        saved = metadata["config"]
        for key in ("image_size", "base_channels", "blocks",
                    "throttle_min", "throttle_max", "map"):
            if saved.get(key) != config.get(key):
                raise ValueError(
                    f"resume configuration mismatch for {key}: "
                    f"checkpoint={saved.get(key)!r}, current={config.get(key)!r}"
                )
        start_epoch = metadata["epoch"] + 1
        train_loader.set_epoch(start_epoch - 1)
    run_id = args.run_name or default_run_id(map_slug(args.map))
    callbacks = CallbackList([] if args.no_monitor_log else [JSONLRunLogger(
        Path(args.runs_root) / run_id, run_id=run_id, device=device
    )])
    monitor_config = {
        **config,
        "device": str(device),
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    callbacks.on_train_begin(
        config=monitor_config,
        data_summary=summarize_records(train_dataset, validation_dataset),
        total_epochs=args.epochs,
    )
    training_started = time.perf_counter()
    step = (start_epoch - 1) * len(train_loader)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            training = train_epoch(
                model, train_loader, optimizer, args.lambda_throttle,
                args.grad_clip_norm,
            )
            step += training["batches"]
            validation = evaluate_model(
                model, validation_loader, args.lambda_throttle
            )
            record = {
                "epoch": epoch,
                "lr": optimizer.lr,
                "train": training,
                "val": validation,
            }
            print(json.dumps(record, sort_keys=True))
            kl.save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                epoch=epoch,
                config=config,
                normalization=train_dataset.normalization,
            )
            config_path = args.run_config or str(Path(args.checkpoint).with_suffix(".json"))
            inference_config = build_inference_config(
                args.checkpoint,
                config,
                train_dataset.normalization,
                epoch=epoch,
                metrics=record,
                config_path=config_path,
            )
            save_inference_config(config_path, inference_config)
            callbacks.on_epoch_end(
                epoch=epoch,
                step=step,
                total_epochs=args.epochs,
                train=training,
                validation=validation,
                lr=optimizer.lr,
                epoch_seconds=training["epoch_seconds"],
                elapsed_seconds=time.perf_counter() - training_started,
            )
    except KeyboardInterrupt:
        callbacks.on_train_end(status="stopped", error="interrupted by user")
        raise
    except BaseException as error:
        callbacks.on_train_end(status="failed", error=error)
        raise
    else:
        callbacks.on_train_end(status="completed")


if __name__ == "__main__":
    main()
