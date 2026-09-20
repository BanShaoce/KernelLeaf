"""Append-only metric monitoring for distributed training."""

from __future__ import annotations

import json
import math
from pathlib import Path


METRIC_FIELDS = (
    "epoch", "step", "data_time", "forward_time", "backward_time",
    "gradient_serialize_time", "gradient_upload_time",
    "parameter_wait_time", "parameter_download_time", "optimizer_time",
    "total_step_time", "samples_per_second", "upload_bytes",
    "download_bytes", "loss", "accuracy", "worker_id",
    "parameter_version",
)


class MetricValidationError(ValueError):
    """Raised when a metric record is incomplete or non-finite."""


def validate_metric(record):
    if not isinstance(record, dict):
        raise MetricValidationError("metric record must be an object")
    missing = sorted(set(METRIC_FIELDS) - set(record))
    if missing:
        raise MetricValidationError(f"metric record is missing fields: {missing}")
    if not isinstance(record["worker_id"], str) or not record["worker_id"]:
        raise MetricValidationError("worker_id must be a non-empty string")
    for name in METRIC_FIELDS:
        if name == "worker_id":
            continue
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricValidationError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise MetricValidationError(f"{name} must be finite and non-negative")
    return dict(record)


class JsonlMonitor:
    """Validate and durably append one JSON object per training step."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)

    def write(self, record):
        value = validate_metric(record)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            file.flush()

    def run(self, metric_queue, expected_records):
        if not isinstance(expected_records, int) or expected_records < 0:
            raise ValueError("expected_records must be a non-negative integer")
        for _ in range(expected_records):
            self.write(metric_queue.get())
