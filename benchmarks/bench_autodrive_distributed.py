"""Benchmark AutoDrive single-process and synchronous PS training."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from apps.autodrive.config import map_slug
from apps.autodrive.train import (
    AutoDriveBenchmarkConfig,
    run_autodrive_ps_benchmark,
    run_autodrive_single_benchmark,
    selected_manifest_map,
)


def _checkpoint_path(output_dir, map_name, mode, timestamp):
    return output_dir / (
        f"autodrive_{map_slug(map_name)}_{mode}_{timestamp}.npz"
    )


def _flat_row(result):
    validation = result["validation"]
    return {
        "mode": result["mode"], "workers": result["workers"],
        "elapsed_seconds": result["elapsed"],
        "samples_per_second": result["samples_per_second"],
        "first_loss": result["first_loss"],
        "last_loss": result["last_loss"],
        "communication_ratio": result["communication_ratio"],
        "val_loss": validation["loss"],
        "steer_mae": validation["steer_mae"],
        "throttle_mae": validation["throttle_mae"],
        "checkpoint": result["checkpoint"],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/DonkeyCar/manifest.jsonl")
    parser.add_argument("--map", default="mountain-track")
    parser.add_argument("--workers", type=int, nargs="+", choices=(1, 2, 4),
                        default=(1, 2, 4))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-throttle", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=60)
    parser.add_argument("--image-width", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--job-timeout", type=float, default=7200.0)
    parser.add_argument("--output-dir", default="bin")
    parser.add_argument("--metrics-dir", default="runs/autodrive_benchmark")
    parser.add_argument("--skip-single-process", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    map_name = selected_manifest_map(args.manifest, args.map)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        manifest=args.manifest, map_name=map_name, epochs=args.epochs,
        global_batch_size=args.global_batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        lambda_throttle=args.lambda_throttle,
        base_channels=args.base_channels,
        image_height=args.image_height, image_width=args.image_width,
        seed=args.seed, device=args.device,
        socket_timeout=args.socket_timeout,
    )
    results = []
    if not args.skip_single_process:
        config = AutoDriveBenchmarkConfig(world_size=1, **common)
        checkpoint = _checkpoint_path(
            output_dir, map_name, "single", timestamp
        )
        results.append(run_autodrive_single_benchmark(config, checkpoint))
    for workers in args.workers:
        config = AutoDriveBenchmarkConfig(world_size=workers, **common)
        mode = f"ps-{workers}w"
        checkpoint = _checkpoint_path(output_dir, map_name, mode, timestamp)
        metrics = metrics_dir / f"{mode}.jsonl"
        results.append(run_autodrive_ps_benchmark(
            config, checkpoint, metrics, timeout=args.job_timeout
        ))

    rows = [_flat_row(result) for result in results]
    summary_base = output_dir / (
        f"autodrive_{map_slug(map_name)}_benchmark_{timestamp}"
    )
    json_path = summary_base.with_suffix(".json")
    csv_path = summary_base.with_suffix(".csv")
    json_path.write_text(json.dumps({
        "timestamp": timestamp, "map": map_name,
        "epochs": args.epochs, "global_batch_size": args.global_batch_size,
        "device": args.device, "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline = rows[0]["samples_per_second"] if (
        rows and rows[0]["mode"] == "single_process"
    ) else None
    print("mode          workers  elapsed_s  samples/s  speedup  comm_%  val_loss  steer_mae  throttle_mae")
    for row in rows:
        speedup = (
            f"{row['samples_per_second'] / baseline:.3f}"
            if baseline is not None else "-"
        )
        print(
            f"{row['mode']:<13} {row['workers']:>7}  "
            f"{row['elapsed_seconds']:>9.3f}  "
            f"{row['samples_per_second']:>9.2f}  {speedup:>7}  "
            f"{100 * row['communication_ratio']:>6.1f}  "
            f"{row['val_loss']:>8.4f}  {row['steer_mae']:>9.4f}  "
            f"{row['throttle_mae']:>12.4f}"
        )
    print(f"summary_json: {json_path}")
    print(f"summary_csv:  {csv_path}")


if __name__ == "__main__":
    main()
