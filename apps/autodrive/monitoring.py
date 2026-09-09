"""JSONL training callbacks and resource snapshots for AutoDrive V11."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

import numpy as np


SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
METRIC_FIELDS = (
    "epoch", "step", "total_epochs", "train_loss", "val_loss",
    "accuracy", "steer_loss", "throttle_loss", "val_steer_loss",
    "val_throttle_loss", "steer_mae", "throttle_mae", "lr",
    "epoch_seconds", "elapsed_seconds", "cpu_percent", "cpu_memory_mb",
    "gpu_memory_mb", "gpu_memory_total_mb",
)


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def validate_run_id(run_id):
    run_id = str(run_id)
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-' (maximum 128 characters)"
        )
    return run_id


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_or_none(name, value):
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric {name!r} must be finite or None")
    return number


def resource_snapshot(device=None):
    """Sample resources without making optional packages mandatory."""
    result = {
        "cpu_percent": None,
        "cpu_memory_mb": None,
        "gpu_memory_mb": None,
        "gpu_memory_total_mb": None,
    }
    try:
        import psutil
        process = psutil.Process()
        result["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        result["cpu_memory_mb"] = process.memory_info().rss / (1024 * 1024)
    except (ImportError, OSError):
        pass
    if device is not None and getattr(device, "kind", None) == "cuda":
        try:
            free_bytes, total_bytes = device.xp.cuda.runtime.memGetInfo()
            result["gpu_memory_mb"] = (total_bytes - free_bytes) / (1024 * 1024)
            result["gpu_memory_total_mb"] = total_bytes / (1024 * 1024)
        except Exception:
            # Monitoring must never interrupt training when a telemetry API is
            # unavailable. The CUDA training path itself still raises normally.
            pass
    return result


def summarize_records(*datasets):
    records = [record for dataset in datasets for record in dataset.records]
    split_counts = Counter(record["split"] for record in records)
    map_counts = {}
    for map_name in sorted({record["map_name"] for record in records}):
        selected = [record for record in records if record["map_name"] == map_name]
        map_counts[map_name] = {
            "total": len(selected),
            "train": sum(record["split"] == "train" for record in selected),
            "val": sum(record["split"] == "val" for record in selected),
        }
    distributions = {}
    for field in ("steering", "throttle"):
        values = np.asarray([float(record[field]) for record in records])
        counts, edges = np.histogram(values, bins=10, range=(-1, 1) if field == "steering" else (0, 1))
        distributions[field] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "counts": counts.tolist(),
            "edges": edges.tolist(),
        }
    return {
        "total": len(records),
        "splits": dict(split_counts),
        "maps": map_counts,
        "distributions": distributions,
    }


class TrainingCallback:
    """Minimal callback protocol; methods intentionally default to no-ops."""

    def on_train_begin(self, **context):
        pass

    def on_epoch_end(self, **context):
        pass

    def on_train_end(self, **context):
        pass


class CallbackList(TrainingCallback):
    def __init__(self, callbacks=()):
        self.callbacks = list(callbacks)

    def _call(self, method, context):
        for callback in self.callbacks:
            getattr(callback, method)(**context)

    def on_train_begin(self, **context):
        self._call("on_train_begin", context)

    def on_epoch_end(self, **context):
        self._call("on_epoch_end", context)

    def on_train_end(self, **context):
        self._call("on_train_end", context)


class JSONLRunLogger(TrainingCallback):
    """Persist one run in a dashboard-readable, append-only directory."""

    def __init__(self, run_dir, run_id=None, device=None):
        self.run_dir = Path(run_dir).resolve()
        self.run_id = validate_run_id(run_id or self.run_dir.name)
        self.device = device
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.started = None

    def on_train_begin(self, *, config, data_summary, total_epochs, **_):
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "gradcam").mkdir()
        self.started = time.perf_counter()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "kind": "autodrive",
            "status": "running",
            "created_at": utc_timestamp(),
            "updated_at": utc_timestamp(),
            "total_epochs": int(total_epochs),
            "config": config,
        }
        _atomic_json(self.run_dir / "run.json", metadata)
        _atomic_json(self.run_dir / "data_summary.json", data_summary)
        self.metrics_path.touch(exist_ok=False)

    def on_epoch_end(self, *, epoch, step, total_epochs, train, validation,
                     lr, epoch_seconds, elapsed_seconds=None, accuracy=None, **_):
        if self.started is None:
            raise RuntimeError("on_train_begin must be called before on_epoch_end")
        resources = resource_snapshot(self.device)
        record = {
            "schema_version": SCHEMA_VERSION,
            "event": "epoch",
            "run_id": self.run_id,
            "timestamp": utc_timestamp(),
            "epoch": int(epoch),
            "step": int(step),
            "total_epochs": int(total_epochs),
            "train_loss": train.get("loss"),
            "val_loss": validation.get("loss"),
            "accuracy": accuracy,
            "steer_loss": train.get("steer_loss"),
            "throttle_loss": train.get("throttle_loss"),
            "val_steer_loss": validation.get("steer_loss"),
            "val_throttle_loss": validation.get("throttle_loss"),
            "steer_mae": validation.get("steer_mae"),
            "throttle_mae": validation.get("throttle_mae"),
            "lr": lr,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": (
                time.perf_counter() - self.started
                if elapsed_seconds is None else elapsed_seconds
            ),
            **resources,
        }
        for field in METRIC_FIELDS:
            if field not in {"epoch", "step", "total_epochs"}:
                record[field] = _finite_or_none(field, record[field])
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()
        _atomic_json(self.run_dir / "latest.json", record)
        _atomic_json(self.run_dir / "validation.json", {
            "epoch": int(epoch),
            "timestamp": record["timestamp"],
            "loss": record["val_loss"],
            "steer_loss": record["val_steer_loss"],
            "throttle_loss": record["val_throttle_loss"],
            "steer_mae": record["steer_mae"],
            "throttle_mae": record["throttle_mae"],
        })
        self._update_status("running", epoch=int(epoch))
        return record

    def _update_status(self, status, **updates):
        path = self.run_dir / "run.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(status=status, updated_at=utc_timestamp(), **updates)
        _atomic_json(path, metadata)

    def on_train_end(self, *, status="completed", error=None, **_):
        if status not in {"completed", "failed", "stopped"}:
            raise ValueError("training status must be completed, failed, or stopped")
        updates = {"finished_at": utc_timestamp()}
        if error:
            updates["error"] = str(error)
        self._update_status(status, **updates)


def default_run_id(map_slug):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return validate_run_id(f"autodrive-{map_slug}-{timestamp}")
