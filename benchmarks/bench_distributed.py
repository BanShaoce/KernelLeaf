"""Measured strong-scaling benchmark for V12.3 distributed training."""

from __future__ import annotations

import argparse
from pathlib import Path

from apps.distributed_mnist import TrainingConfig, run_distributed_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2, 4),
                        choices=(1, 2, 4))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default="runs/distributed_benchmark")
    args = parser.parse_args(argv)

    rows = []
    for workers in args.workers:
        config = TrainingConfig(
            world_size=workers, epochs=args.epochs,
            global_batch_size=args.global_batch_size,
            synthetic_samples=args.samples, device=args.device,
            hidden_sizes=(64, 32),
        )
        output = Path(args.output_dir) / f"workers-{workers}.jsonl"
        summary = run_distributed_training(config, output, timeout=1800.0)
        rows.append(summary)

    print("workers  elapsed_s  samples/s  first_loss  last_loss")
    for row in rows:
        print(
            f"{row['world_size']:>7}  {row['elapsed']:>9.3f}  "
            f"{row['samples_per_second']:>9.2f}  "
            f"{row['first_loss']:>10.4f}  {row['last_loss']:>9.4f}"
        )


if __name__ == "__main__":
    main()
