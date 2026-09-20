"""Run lightweight multi-node sparse LR and LDA parameter-server experiments."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from typing import Dict, Iterable, Mapping, Sequence

from .runtime import LocalCluster, ParameterServerClient

from .lda import LDAConfig, evaluate_lda, run_lda_worker
from .sparse_lr import (
    SparseLRConfig,
    evaluate_sparse_lr,
    run_sparse_lr_worker,
)


def _collect_worker_results(queue, count: int, timeout: float):
    results = []
    deadline = time.monotonic() + timeout
    for _ in range(count):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for worker results")
        result = queue.get(timeout=remaining)
        if "error" in result:
            raise RuntimeError(
                f"worker {result.get('worker')} failed: {result['error']}\n"
                f"{result.get('traceback', '')}"
            )
        results.append(result)
    return results


def summarize_worker_epochs(
    worker_results,
    *,
    warmup_epochs: int = 0,
) -> Dict[str, float]:
    """Average per-epoch wall, compute, communication, and payload metrics."""
    epoch_rows = [result.get("epoch_metrics", ()) for result in worker_results]
    if not epoch_rows or not epoch_rows[0]:
        raise ValueError("worker results do not contain epoch metrics")
    common_epochs = min(len(rows) for rows in epoch_rows)
    if warmup_epochs < 0 or warmup_epochs >= common_epochs:
        raise ValueError(
            f"warmup_epochs must be in [0, {common_epochs}), got {warmup_epochs}"
        )
    summaries = []
    for epoch in range(warmup_epochs, common_epochs):
        rows = [worker_epochs[epoch] for worker_epochs in epoch_rows]
        summaries.append(
            {
                "total_time": max(float(row["wall_seconds"]) for row in rows),
                "compute_time": sum(float(row["compute_seconds"]) for row in rows),
                "communication_time": sum(
                    float(row["communication_seconds"]) for row in rows
                ),
                "message_bytes": sum(int(row["message_bytes"]) for row in rows),
            }
        )
    count = float(len(summaries))
    return {
        "one_epoch_total_time": sum(row["total_time"] for row in summaries) / count,
        "compute_time": sum(row["compute_time"] for row in summaries) / count,
        "communication_time": sum(
            row["communication_time"] for row in summaries
        ) / count,
        "message_bytes": sum(row["message_bytes"] for row in summaries) / count,
        "epochs_measured": int(count),
    }


def run_sparse_lr_experiment(
    *,
    transport: str,
    mode: str,
    workers: int,
    shards: int,
    config: SparseLRConfig,
    timeout: float = 60.0,
    worker_cpu_affinity=None,
    server_cpu_affinity=None,
) -> Dict[str, object]:
    cluster = LocalCluster(
        num_shards=shards,
        workers=workers,
        transport=transport,
        mode=mode,
        barrier_timeout=timeout,
        startup_timeout=timeout,
        server_cpu_affinity=server_cpu_affinity,
    )
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    registration_barrier = context.Barrier(workers)
    processes = []
    started = time.perf_counter()
    with cluster:
        for worker_id in range(workers):
            process = context.Process(
                target=run_sparse_lr_worker,
                args=(cluster.endpoints, worker_id),
                kwargs={
                    "workers": workers,
                    "transport": transport,
                    "config": config,
                    "timeout": timeout,
                    "result_queue": result_queue,
                    "registration_barrier": registration_barrier,
                    "cpu_affinity": (
                        None
                        if worker_cpu_affinity is None
                        else worker_cpu_affinity[worker_id]
                    ),
                },
                name=f"kernelleaf-lr-{worker_id}",
            )
            process.start()
            processes.append(process)
        worker_results = _collect_worker_results(
            result_queue, workers, timeout * max(1, config.epochs)
        )
        for process in processes:
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                raise TimeoutError(f"sparse-LR worker {process.name} did not stop")
            if process.exitcode not in (0, None):
                raise RuntimeError(
                    f"sparse-LR worker {process.name} exited with {process.exitcode}"
                )

        client = ParameterServerClient(
            cluster.endpoints, worker_id="lr-evaluator", transport=transport, timeout=timeout
        )
        try:
            metrics = evaluate_sparse_lr(client, config=config)
            server_metadata = client.metadata()
        finally:
            client.close()
    return {
        "experiment": "sparse_logistic_regression",
        "transport": transport,
        "requested_mode": mode,
        "workers": workers,
        "shards": shards,
        "elapsed": time.perf_counter() - started,
        "worker_results": worker_results,
        "server_metadata": server_metadata,
        "epoch_summary": summarize_worker_epochs(worker_results),
        **metrics,
    }


def run_lda_experiment(
    *,
    transport: str,
    mode: str,
    workers: int,
    shards: int,
    config: LDAConfig,
    timeout: float = 60.0,
) -> Dict[str, object]:
    cluster = LocalCluster(
        num_shards=shards,
        workers=workers,
        transport=transport,
        mode=mode,
        barrier_timeout=timeout,
        startup_timeout=timeout,
    )
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    registration_barrier = context.Barrier(workers)
    processes = []
    started = time.perf_counter()
    with cluster:
        for worker_id in range(workers):
            process = context.Process(
                target=run_lda_worker,
                args=(cluster.endpoints, worker_id),
                kwargs={
                    "workers": workers,
                    "transport": transport,
                    "config": config,
                    "timeout": timeout,
                    "result_queue": result_queue,
                    "registration_barrier": registration_barrier,
                },
                name=f"kernelleaf-lda-{worker_id}",
            )
            process.start()
            processes.append(process)
        worker_results = _collect_worker_results(
            result_queue, workers, timeout * max(1, config.sweeps)
        )
        for process in processes:
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                raise TimeoutError(f"LDA worker {process.name} did not stop")
            if process.exitcode not in (0, None):
                raise RuntimeError(
                    f"LDA worker {process.name} exited with {process.exitcode}"
                )

        client = ParameterServerClient(
            cluster.endpoints, worker_id="lda-evaluator", transport=transport, timeout=timeout
        )
        try:
            metrics = evaluate_lda(client, worker_results, config=config)
            server_metadata = client.metadata()
        finally:
            client.close()
    serializable_workers = [
        {key: value for key, value in result.items() if key not in {"assignments", "doc_counts"}}
        for result in worker_results
    ]
    return {
        "experiment": "latent_dirichlet_allocation",
        "transport": transport,
        "requested_mode": mode,
        "workers": workers,
        "shards": shards,
        "elapsed": time.perf_counter() - started,
        "worker_results": serializable_workers,
        "server_metadata": server_metadata,
        **metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("lr", "lda", "all"), default="lr")
    parser.add_argument("--transport", choices=("socket", "grpc"), default="socket")
    parser.add_argument("--mode", choices=("sync", "async"), default="sync")
    parser.add_argument(
        "--switch-at",
        type=int,
        default=-1,
        help="round/sweep index for a runtime consistency switch; -1 disables it",
    )
    parser.add_argument("--switch-to", choices=("sync", "async"), default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)

    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--features", type=int, default=4096)
    parser.add_argument("--nnz", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.8)

    parser.add_argument("--documents", type=int, default=320)
    parser.add_argument("--vocabulary", type=int, default=256)
    parser.add_argument("--topics", type=int, default=4)
    parser.add_argument("--doc-length", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worker-delay", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.workers <= 0 or args.shards <= 0:
        raise SystemExit("--workers and --shards must be positive")
    if args.switch_to and args.switch_at < 0:
        raise SystemExit("--switch-to requires a non-negative --switch-at")

    outputs = []
    if args.experiment in {"lr", "all"}:
        lr_config = SparseLRConfig(
            samples=args.samples,
            features=args.features,
            nnz=args.nnz,
            validation_samples=args.validation_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            worker_delay=args.worker_delay,
            mode=args.mode,
            switch_at=args.switch_at,
            switch_to=args.switch_to,
        )
        outputs.append(
            run_sparse_lr_experiment(
                transport=args.transport,
                mode=args.mode,
                workers=args.workers,
                shards=args.shards,
                config=lr_config,
                timeout=args.timeout,
            )
        )
    if args.experiment in {"lda", "all"}:
        lda_config = LDAConfig(
            documents=args.documents,
            vocabulary=args.vocabulary,
            topics=args.topics,
            doc_length=args.doc_length,
            sweeps=args.sweeps,
            alpha=args.alpha,
            beta=args.beta,
            seed=args.seed,
            worker_delay=args.worker_delay,
            mode=args.mode,
            switch_at=args.switch_at,
            switch_to=args.switch_to,
        )
        outputs.append(
            run_lda_experiment(
                transport=args.transport,
                mode=args.mode,
                workers=args.workers,
                shards=args.shards,
                config=lda_config,
                timeout=args.timeout,
            )
        )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
