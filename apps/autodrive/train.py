"""Train the KernelLeaf dual-head AutoDrive ResNet with V11 telemetry."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import kernelleaf as kl
import kernelleaf.nn as nn
from kernelleaf.data import DataLoader

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
