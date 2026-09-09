import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from apps.autodrive.demo_dashboard import generate
from apps.autodrive.monitoring import (
    CallbackList, JSONLRunLogger, METRIC_FIELDS, TrainingCallback,
    summarize_records, validate_run_id,
)


class _Dataset:
    def __init__(self, records):
        self.records = records


def _summary():
    train = _Dataset([
        {"split": "train", "map_name": "warren-track", "steering": -0.5, "throttle": 0.2},
        {"split": "train", "map_name": "warren-track", "steering": 0.5, "throttle": 0.3},
    ])
    validation = _Dataset([
        {"split": "val", "map_name": "warren-track", "steering": 0.0, "throttle": 0.25},
    ])
    return summarize_records(train, validation)


def _epoch(logger, epoch=1):
    return logger.on_epoch_end(
        epoch=epoch,
        step=epoch * 2,
        total_epochs=2,
        train={"loss": 0.4, "steer_loss": 0.3, "throttle_loss": 0.1},
        validation={"loss": 0.5, "steer_loss": 0.35, "throttle_loss": 0.15,
                    "steer_mae": 0.2, "throttle_mae": 0.08},
        lr=0.001,
        epoch_seconds=1.2,
        elapsed_seconds=1.2,
    )


def test_jsonl_logger_schema_state_and_dataset_summary(tmp_path):
    logger = JSONLRunLogger(tmp_path / "run-one", device=None)
    summary = _summary()
    logger.on_train_begin(
        config={"map": "warren-track"}, data_summary=summary, total_epochs=2
    )
    record = _epoch(logger)
    logger.on_train_end(status="completed")

    assert set(METRIC_FIELDS) <= set(record)
    assert record["schema_version"] == 1
    assert record["accuracy"] is None
    assert json.loads((logger.run_dir / "run.json").read_text())["status"] == "completed"
    assert json.loads((logger.run_dir / "latest.json").read_text())["epoch"] == 1
    assert json.loads((logger.run_dir / "validation.json").read_text())["steer_mae"] == 0.2
    assert summary["total"] == 3
    assert summary["splits"] == {"train": 2, "val": 1}
    assert summary["maps"]["warren-track"] == {"total": 3, "train": 2, "val": 1}


def test_logger_rejects_unsafe_names_duplicate_runs_and_nonfinite_metrics(tmp_path):
    for value in ("../escape", "has space", "", "a/child"):
        with pytest.raises(ValueError):
            validate_run_id(value)
    logger = JSONLRunLogger(tmp_path / "safe-run")
    logger.on_train_begin(config={}, data_summary=_summary(), total_epochs=2)
    with pytest.raises(FileExistsError):
        JSONLRunLogger(tmp_path / "safe-run").on_train_begin(
            config={}, data_summary=_summary(), total_epochs=2
        )
    with pytest.raises(ValueError, match="finite"):
        logger.on_epoch_end(
            epoch=1, step=1, total_epochs=2,
            train={"loss": np.inf, "steer_loss": 0.1, "throttle_loss": 0.1},
            validation={"loss": 0.2}, lr=0.01, epoch_seconds=1,
        )


def test_callback_list_preserves_training_when_empty_and_forwards_events():
    calls = []

    class Recorder(TrainingCallback):
        def on_epoch_end(self, **context):
            calls.append(context["epoch"])

    CallbackList().on_epoch_end(epoch=1)
    callbacks = CallbackList([Recorder(), Recorder()])
    callbacks.on_epoch_end(epoch=3)
    assert calls == [3, 3]


def test_training_import_does_not_require_dashboard_runtime():
    subprocess.run([
        sys.executable,
        "-c",
        "import sys; import apps.autodrive.train; "
        "assert 'fastapi' not in sys.modules; assert 'uvicorn' not in sys.modules",
    ], check=True)


def test_demo_generator_creates_complete_readable_run(tmp_path):
    run_dir = generate(tmp_path, "demo-run", epochs=3)
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[-1])["epoch"] == 3
    assert json.loads((run_dir / "run.json").read_text())["status"] == "completed"


def test_dashboard_api_endpoints_and_corrupt_log_recovery(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from apps.autodrive.dashboard_api import create_app

    run_dir = generate(tmp_path, "dashboard-run", epochs=2)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write("{broken json\n")
        file.write("{}\n")
    (run_dir / "gradcam" / "steering.png").write_bytes(
        b"\x89PNG\r\n\x1a\nsynthetic"
    )
    broken = tmp_path / "broken-run"
    broken.mkdir()
    (broken / "run.json").write_text("not json", encoding="utf-8")

    client = TestClient(create_app(tmp_path, frontend_dir=tmp_path / "absent"))
    assert client.get("/api/health").status_code == 200
    runs = client.get("/api/runs").json()["runs"]
    assert {run["run_id"] for run in runs} == {"dashboard-run", "broken-run"}
    assert next(run for run in runs if run["run_id"] == "broken-run")["status"] == "invalid"
    metrics = client.get("/api/runs/dashboard-run/metrics").json()
    assert len(metrics["metrics"]) == 2
    assert [error["line"] for error in metrics["errors"]] == [3, 4]
    assert client.get("/api/runs/dashboard-run/latest").json()["latest"]["epoch"] == 2
    assert client.get("/api/runs/dashboard-run/validation").json()["validation"]["epoch"] == 2
    assert client.get("/api/runs/dashboard-run/data-summary").json()["summary"]["total"] == 10000
    status = client.get("/api/train/status").json()
    assert status["run_id"] == "dashboard-run"
    assert status["epoch"] == 2
    assert status["running"] is False
    history = client.get("/api/train/history").json()["epochs"]
    assert len(history) == 2
    assert history[-1]["n_batches"] == 250
    assert client.get("/api/train/config").json()["batch_size"] == 32
    assert len(client.get("/api/system/history").json()["samples"]) == 2
    assert client.get("/api/model/info").json()["architecture"] == "AutoDriveResNet"
    assert client.get("/api/model/checkpoints").json()["checkpoints"] == []
    assert client.get("/api/eval/validation").json()["n_samples"] == 2000
    assert client.get("/api/data/info").json()["train"]["count"] == 8000
    assert client.get("/api/data/files").json()["files"] == []
    images = client.get("/api/runs/dashboard-run/gradcam").json()["images"]
    assert images[0]["name"] == "steering.png"
    assert client.get(images[0]["url"]).status_code == 200
    with client.websocket_connect("/ws/runs/dashboard-run") as websocket:
        assert websocket.receive_json()["latest"]["epoch"] == 2
    with client.websocket_connect("/ws/train") as websocket:
        assert websocket.receive_json()["epoch"] == 2
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws/runs/dashboard-run", headers={"origin": "https://evil.example"}
        ):
            pass
    assert client.get("/api/runs/missing-run").status_code == 404
    assert client.get("/api/runs/%2E%2E%2Fescape").status_code in {400, 404}


def test_dashboard_missing_optional_files_return_structured_errors(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from apps.autodrive.dashboard_api import create_app

    run = tmp_path / "empty-run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"run_id": "empty-run"}))
    client = TestClient(create_app(tmp_path, frontend_dir=tmp_path / "absent"))
    assert client.get("/api/runs/empty-run/metrics").json()["errors"]
    assert client.get("/api/runs/empty-run/latest").json()["latest"] is None
    assert client.get("/api/runs/empty-run/validation").json()["error"]
    assert client.get("/api/runs/empty-run/data-summary").json()["error"]


def test_dashboard_starts_real_training_command_and_validates_form(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from apps.autodrive.dashboard_api import create_app

    workspace = tmp_path / "workspace"
    data_root = workspace / "data" / "DonkeyCar"
    data_root.mkdir(parents=True)
    manifest = data_root / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"map_name": "warren", "split": "train"}) + "\n" +
        json.dumps({"map_name": "mountain", "split": "val"}) + "\n",
        encoding="utf-8",
    )

    class FakeController:
        def __init__(self):
            self.started = None
            self.state = {
                "running": False, "paused": False, "run_id": None,
                "config": {}, "message": "idle",
            }

        def snapshot(self):
            return dict(self.state)

        def start(self, command, run_id, config, cwd):
            self.started = (command, run_id, config, cwd)
            self.state.update(
                running=True, run_id=run_id, config=config, message="正在启动"
            )

        def pause(self):
            self.state.update(paused=True, message="训练已暂停")

        def resume(self):
            self.state.update(paused=False, message="训练已恢复")

        def stop(self):
            self.state.update(running=False, paused=False, message="正在停止")
            return True

    app = create_app(
        tmp_path / "runs", frontend_dir=tmp_path / "absent",
        project_root=workspace, data_root=data_root,
    )
    controller = FakeController()
    app.state.training = controller
    client = TestClient(app)
    response = client.post("/api/train/start", json={
        "manifest": "data/DonkeyCar/manifest.jsonl",
        "map": "warren",
        "device": "cpu", "epochs": 3, "batch_size": 8,
        "lr": 0.002, "weight_decay": 0.0001, "grad_clip_norm": 2,
    })
    assert response.status_code == 200
    command, run_id, config, cwd = controller.started
    assert command[:4] == [sys.executable, "-m", "apps.autodrive", "train"]
    assert command[command.index("--map") + 1] == "warren"
    assert config["map"] == "warren"
    assert config["grad_clip_norm"] == 2
    assert cwd == workspace
    assert client.get("/api/train/status").json()["run_id"] == run_id
    assert client.post("/api/train/pause").status_code == 200
    assert client.post("/api/train/resume").status_code == 200
    assert client.post("/api/train/stop").status_code == 200

    rejected = client.post("/api/train/start", json={
        "manifest": "data/DonkeyCar/manifest.jsonl",
        "maps": "warren, mountain",
    })
    assert rejected.status_code == 422
    assert "不再支持 maps" in rejected.json()["detail"]

    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    invalid = client.post("/api/train/start", json={"manifest": str(outside)})
    assert invalid.status_code == 422


def test_autodrive_training_writes_dashboard_run(tmp_path, monkeypatch):
    from apps.autodrive import train
    from apps.autodrive.manifest import write_manifest

    records = []
    for index in range(6):
        image = tmp_path / f"frame-{index}.png"
        Image.fromarray(np.full((10, 12, 3), 30 + index, dtype=np.uint8)).save(image)
        records.append({
            "image_path": image.name,
            "steering": (index - 3) / 6,
            "throttle": 0.2,
            "map_name": "synthetic",
            "run_id": "train" if index < 4 else "val",
            "split": "train" if index < 4 else "val",
        })
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    checkpoint = tmp_path / "model.npz"
    runs_root = tmp_path / "runs"
    monkeypatch.setattr("sys.argv", [
        "train", "--manifest", str(manifest), "--map", "synthetic",
        "--epochs", "1", "--batch-size", "4", "--base-channels", "2",
        "--image-height", "8", "--image-width", "10", "--num-workers", "0",
        "--checkpoint", str(checkpoint), "--runs-root", str(runs_root),
        "--run-name", "integration-run",
    ])

    train.main()

    run = runs_root / "integration-run"
    assert json.loads((run / "run.json").read_text())["status"] == "completed"
    assert json.loads((run / "latest.json").read_text())["epoch"] == 1
    assert checkpoint.is_file()
