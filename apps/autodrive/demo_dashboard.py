"""Generate a small synthetic V11 dashboard run without training a model."""

import argparse
from pathlib import Path

import numpy as np

from .monitoring import JSONLRunLogger, default_run_id


def generate(output_root="runs", run_id=None, epochs=20, seed=7):
    run_id = run_id or default_run_id("demo")
    logger = JSONLRunLogger(Path(output_root) / run_id, run_id=run_id)
    logger.on_train_begin(
        config={"map": "warren-track", "device": "demo", "epochs": epochs,
                "batch_size": 32, "lr": 0.001, "weight_decay": 0.0001},
        data_summary={
            "total": 10000,
            "splits": {"train": 8000, "val": 2000},
            "maps": {"warren-track": {"total": 10000, "train": 8000, "val": 2000}},
            "distributions": {
                "steering": {"min": -1, "max": 1, "mean": 0.01, "std": 0.32,
                             "counts": [180, 320, 640, 1100, 2600, 2700, 1200, 650, 390, 220],
                             "edges": np.linspace(-1, 1, 11).tolist()},
                "throttle": {"min": 0.1, "max": 0.5, "mean": 0.24, "std": 0.08,
                             "counts": [0, 900, 6100, 2500, 500, 0, 0, 0, 0, 0],
                             "edges": np.linspace(0, 1, 11).tolist()},
            },
        },
        total_epochs=epochs,
    )
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        decay = np.exp(-epoch / 5)
        train = {"loss": 0.55 * decay + 0.018, "steer_loss": 0.4 * decay + 0.012,
                 "throttle_loss": 0.15 * decay + 0.006}
        validation = {"loss": 0.62 * decay + 0.026 + rng.uniform(0, 0.004),
                      "steer_loss": 0.44 * decay + 0.017,
                      "throttle_loss": 0.18 * decay + 0.009,
                      "steer_mae": 0.31 * decay + 0.045,
                      "throttle_mae": 0.12 * decay + 0.018}
        logger.on_epoch_end(
            epoch=epoch, step=epoch * 250, total_epochs=epochs,
            train=train, validation=validation, lr=0.001,
            epoch_seconds=2.8 + rng.uniform(-0.2, 0.2),
            elapsed_seconds=epoch * 2.8,
        )
    logger.on_train_end(status="completed")
    return logger.run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(generate(args.output_root, args.run_id, args.epochs, args.seed))


if __name__ == "__main__":
    main()
