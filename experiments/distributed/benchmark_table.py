"""Run the four configurations required by the parameter-server result table."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Dict, Sequence

from .runtime.cpu_affinity import (
    cpu_model_name,
    format_core_groups,
    physical_core_groups,
    pinned_to_cpus,
)

from .run_experiments import run_sparse_lr_experiment, summarize_worker_epochs
from .sparse_lr import SparseLRConfig, run_sparse_lr_baseline


TABLE_CONFIGURATIONS = (
    ("single_baseline", "单机（基线）", "none", 1, 0),
    ("socket_json_2", "Socket+JSON 2 Worker", "socket", 2, 2),
    ("grpc_protobuf_2", "gRPC+protobuf 2 Worker", "grpc", 2, 2),
    ("socket_json_4", "Socket+JSON 4 Worker", "socket", 4, 2),
)


def _mean(values):
    values = list(values)
    return sum(values) / float(len(values))


def _run_configuration(
    *,
    label: str,
    transport: str,
    workers: int,
    shards: int,
    config: SparseLRConfig,
    repeats: int,
    warmup_epochs: int,
    timeout: float,
    core_groups=None,
    quiet: bool = False,
) -> Dict[str, object]:
    runs = []
    for repeat in range(repeats):
        if not quiet:
            affinity_text = (
                "unpinned"
                if core_groups is None
                else (
                    format_core_groups(core_groups[:workers])
                    + " workers; "
                    + format_core_groups(
                        core_groups[workers : workers + max(1, shards)]
                    )
                    + " servers"
                )
            )
            print(
                f"[{label}] repeat {repeat + 1}/{repeats}; CPU={affinity_text}",
                file=sys.stderr,
            )
        if transport == "none":
            if core_groups is None:
                result = run_sparse_lr_baseline(config=config)
            else:
                with pinned_to_cpus(core_groups[0]):
                    result = run_sparse_lr_baseline(config=config)
        else:
            result = run_sparse_lr_experiment(
                transport=transport,
                mode=config.mode,
                workers=workers,
                shards=shards,
                config=config,
                timeout=timeout,
                worker_cpu_affinity=(
                    None if core_groups is None else core_groups[:workers]
                ),
                server_cpu_affinity=(
                    None
                    if core_groups is None
                    else core_groups[workers : workers + shards]
                ),
            )
        summary = summarize_worker_epochs(
            result["worker_results"], warmup_epochs=warmup_epochs
        )
        runs.append(
            {
                **summary,
                "accuracy": float(result["accuracy"]),
                "log_loss": float(result["log_loss"]),
                "repeat": repeat,
            }
        )
    return {
        "configuration": label,
        "transport": transport,
        "workers": workers,
        "shards": shards,
        "one_epoch_total_time": _mean(
            run["one_epoch_total_time"] for run in runs
        ),
        "compute_time": _mean(run["compute_time"] for run in runs),
        "communication_time": _mean(
            run["communication_time"] for run in runs
        ),
        "message_bytes": _mean(run["message_bytes"] for run in runs),
        "accuracy": _mean(run["accuracy"] for run in runs),
        "log_loss": _mean(run["log_loss"] for run in runs),
        "repeats": repeats,
        "warmup_epochs": warmup_epochs,
        "raw_runs": runs,
    }


def _markdown(rows) -> str:
    lines = [
        "| 配置 | 一个 epoch 总时间 (s) | 计算用时 (s) | 通信用时 (s) | 消息字节 (B) | 准确率 | 加速比 | 理想加速比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {configuration} | {one_epoch_total_time:.6f} | "
            "{compute_time:.6f} | {communication_time:.6f} | "
            "{message_bytes:.0f} | {accuracy:.4f} | {speedup:.3f} | "
            "{ideal_speedup:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "说明：总时间取每个 epoch 所有 worker 的最长墙钟时间；计算和通信用时为各 worker 累加值；消息字节为序列化 payload 的请求+响应字节，不含 TCP/HTTP2 framing；加速比=单机总时间/当前配置总时间。",
        ]
    )
    return "\n".join(lines)


def _csv(rows) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "configuration",
            "one_epoch_total_time_s",
            "compute_time_s",
            "communication_time_s",
            "message_bytes",
            "accuracy",
            "speedup",
            "ideal_speedup",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row["configuration"],
                f"{row['one_epoch_total_time']:.6f}",
                f"{row['compute_time']:.6f}",
                f"{row['communication_time']:.6f}",
                f"{row['message_bytes']:.0f}",
                f"{row['accuracy']:.6f}",
                f"{row['speedup']:.6f}",
                f"{row['ideal_speedup']:.1f}",
            ]
        )
    return output.getvalue().rstrip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--no-pin-cpu",
        action="store_true",
        help="disable physical-core affinity control (not recommended for speedup data)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--skip-grpc",
        action="store_true",
        help="omit the gRPC row instead of failing when grpcio is unavailable",
    )

    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--nnz", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise SystemExit("--warmup-epochs must be in [0, epochs)")
    config = SparseLRConfig(
        samples=args.samples,
        features=args.features,
        nnz=args.nnz,
        validation_samples=args.validation_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        mode="sync",
    )
    core_groups = None if args.no_pin_cpu else physical_core_groups()
    if core_groups is not None:
        required_cores = max(workers for _, _, _, workers, _ in TABLE_CONFIGURATIONS)
        required_cores += max(shards for _, _, _, _, shards in TABLE_CONFIGURATIONS)
        if len(core_groups) < required_cores:
            raise SystemExit(
                f"benchmark needs {required_cores} physical cores for isolated "
                f"workers and servers, found {len(core_groups)}"
            )
        print(
            f"CPU: {cpu_model_name()}; physical cores: "
            f"{format_core_groups(core_groups)}",
            file=sys.stderr,
        )
    rows = []
    for _, label, transport, workers, shards in TABLE_CONFIGURATIONS:
        if transport == "grpc" and args.skip_grpc:
            continue
        rows.append(
            _run_configuration(
                label=label,
                transport=transport,
                workers=workers,
                shards=shards,
                config=config,
                repeats=args.repeat,
                warmup_epochs=args.warmup_epochs,
                timeout=args.timeout,
                core_groups=core_groups,
                quiet=args.quiet,
            )
        )
    baseline_time = float(rows[0]["one_epoch_total_time"])
    for row in rows:
        row["speedup"] = baseline_time / float(row["one_epoch_total_time"])
        row["ideal_speedup"] = float(row["workers"])
    if args.format == "json":
        rendered = json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True)
    elif args.format == "csv":
        rendered = _csv(rows)
    else:
        rendered = _markdown(rows)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print(destination.resolve())
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
