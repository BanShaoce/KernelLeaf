"""Measured strong-scaling benchmark for V12.3 distributed training."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import kernelleaf as kl
import kernelleaf.nn as nn

from apps.distributed_mnist import (
    TrainingConfig,
    _device,
    _global_batches,
    _load_training_arrays,
    _make_model,
    _synchronize,
    run_distributed_training,
)


def run_single_process_baseline(config):
    """Train without PS, sockets, spawned Workers, or a Monitor process."""
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig")
    started = time.perf_counter()
    images, labels = _load_training_arrays(config)
    device = _device(config)
    model = _make_model(config)
    model.train()
    optimizer = kl.optim.SGD(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SoftmaxLoss()
    losses = []
    steps = 0
    for epoch in range(config.epochs):
        for indices in _global_batches(len(images), config, epoch):
            optimizer.reset_grad()
            inputs = kl.Tensor(images[indices], device=device)
            targets = kl.Tensor(labels[indices], device=device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss_value = float(loss.numpy())
            loss.backward()
            optimizer.step()
            losses.append(loss_value)
            steps += 1
    # Include the final queued optimizer work in CUDA end-to-end timing.
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "mode": "single_process",
        "world_size": 0,
        "steps_per_epoch": steps // config.epochs,
        "elapsed": elapsed,
        "samples_per_second": (len(images) * config.epochs) / elapsed,
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2, 4),
                        choices=(1, 2, 4))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", choices=("mlp", "lenet5"), default="mlp")
    parser.add_argument("--output-dir", default="runs/distributed_benchmark")
    parser.add_argument(
        "--skip-single-process", action="store_true",
        help="omit the pure single-process reference row",
    )
    args = parser.parse_args(argv)

    rows = []
    common = dict(
        epochs=args.epochs,
        global_batch_size=args.global_batch_size,
        synthetic_samples=args.samples,
        device=args.device,
        hidden_sizes=(64, 32),
        model_name=args.model,
    )
    if not args.skip_single_process:
        # world_size only affects distributed sharding; 1 keeps batch validation
        # valid while the baseline itself launches no distributed roles.
        rows.append(run_single_process_baseline(TrainingConfig(
            world_size=1, **common
        )))
    for workers in args.workers:
        config = TrainingConfig(world_size=workers, **common)
        output = Path(args.output_dir) / f"{args.model}-workers-{workers}.jsonl"
        summary = run_distributed_training(config, output, timeout=1800.0)
        summary["mode"] = f"ps_{workers}_worker" + ("s" if workers != 1 else "")
        rows.append(summary)

    baseline_throughput = rows[0]["samples_per_second"] if (
        rows and rows[0]["mode"] == "single_process"
    ) else None
    print(f"model: {args.model}")
    print("mode             workers  elapsed_s  samples/s  speedup  first_loss  last_loss")
    for row in rows:
        workers = "-" if row["mode"] == "single_process" else row["world_size"]
        speedup = (
            f"{row['samples_per_second'] / baseline_throughput:.3f}"
            if baseline_throughput is not None else "-"
        )
        print(
            f"{row['mode']:<16}  {str(workers):>7}  {row['elapsed']:>9.3f}  "
            f"{row['samples_per_second']:>9.2f}  {speedup:>7}  "
            f"{row['first_loss']:>10.4f}  {row['last_loss']:>9.4f}"
        )


if __name__ == "__main__":
    main()
