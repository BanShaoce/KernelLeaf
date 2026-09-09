"""Read-only FastAPI dashboard for KernelLeaf AutoDrive training runs."""

import argparse
import asyncio
from datetime import datetime
import json
import math
import mimetypes
from pathlib import Path
import subprocess
import sys
import threading

from .monitoring import METRIC_FIELDS, SCHEMA_VERSION, default_run_id, validate_run_id


class TrainingProcessController:
    """Own the single training subprocess started by this dashboard process."""

    def __init__(self, run_root):
        self.run_root = Path(run_root)
        self.lock = threading.Lock()
        self.process = None
        self.run_id = None
        self.config = {}
        self.paused = False
        self.message = "idle"
        self.stop_requested = False
        self.console_path = None
        self.console_file = None

    def snapshot(self):
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            return {
                "running": running, "paused": running and self.paused,
                "run_id": self.run_id, "config": dict(self.config),
                "message": self.message,
                "console_log": str(self.console_path) if self.console_path else None,
            }

    def start(self, command, run_id, config, cwd):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("训练已在运行中")
            self.run_id = run_id
            self.config = dict(config)
            self.paused = False
            self.stop_requested = False
            self.message = "正在启动"
            self.run_root.mkdir(parents=True, exist_ok=True)
            self.console_path = self.run_root / f"{run_id}.console.log"
            self.console_file = self.console_path.open("w", encoding="utf-8")
            try:
                self.process = subprocess.Popen(
                    command, cwd=str(cwd), stdin=subprocess.DEVNULL,
                    stdout=self.console_file, stderr=subprocess.STDOUT,
                )
            except BaseException:
                self.console_file.close()
                self.console_file = None
                raise
            process = self.process
        threading.Thread(
            target=self._wait, args=(process,), daemon=True,
            name="kernelleaf-autodrive-dashboard-waiter",
        ).start()

    def _wait(self, process):
        return_code = process.wait()
        with self.lock:
            if process is not self.process:
                return
            self.paused = False
            if self.console_file:
                self.console_file.close()
                self.console_file = None
            failure_detail = ""
            if return_code != 0 and self.console_path and self.console_path.is_file():
                try:
                    lines = self.console_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    failure_detail = next((line.strip() for line in reversed(lines) if line.strip()), "")
                except OSError:
                    pass
            self.message = (
                "训练已停止" if self.stop_requested else
                "训练完成" if return_code == 0 else
                f"训练失败（退出码 {return_code}）：{failure_detail or '请检查 console log'}"
            )
            run_id = self.run_id
            stopped = self.stop_requested
        metadata_path = self.run_root / run_id / "run.json" if run_id else None
        if metadata_path and metadata_path.is_file():
            try:
                metadata = _read_json(metadata_path)
                if metadata.get("status") == "running":
                    metadata["status"] = "stopped" if stopped else "failed"
                    metadata["error"] = self.message
                    temporary = metadata_path.with_suffix(".json.tmp")
                    temporary.write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    temporary.replace(metadata_path)
            except (OSError, json.JSONDecodeError):
                pass

    def pause(self):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("当前没有正在运行的训练")
            if not self.paused:
                import psutil
                psutil.Process(self.process.pid).suspend()
                self.paused = True
                self.message = "训练已暂停"

    def resume(self):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("当前没有正在运行的训练")
            if self.paused:
                import psutil
                psutil.Process(self.process.pid).resume()
                self.paused = False
                self.message = "训练已恢复"

    def stop(self):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                return False
            if self.paused:
                import psutil
                psutil.Process(self.process.pid).resume()
                self.paused = False
            self.stop_requested = True
            self.message = "正在停止"
            self.process.terminate()
            return True


def _dashboard_dependencies():
    try:
        from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as error:
        raise RuntimeError(
            "dashboard support requires `pip install -e '.[dashboard]'`"
        ) from error
    return (FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect,
            CORSMiddleware, FileResponse, StaticFiles)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_metrics(path, limit=None):
    records, errors = [], []
    if not path.is_file():
        return records, [{"line": None, "error": "metrics.jsonl is missing"}]
    with path.open("r", encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be an object")
                if value.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("unsupported or missing schema_version")
                if value.get("event") != "epoch":
                    raise ValueError("record event must be 'epoch'")
                missing = sorted(set(METRIC_FIELDS) - set(value))
                if missing:
                    raise ValueError(f"record is missing fields: {missing}")
                for field in METRIC_FIELDS:
                    metric = value[field]
                    if metric is not None and not math.isfinite(float(metric)):
                        raise ValueError(f"metric {field!r} is not finite")
                records.append(value)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                errors.append({"line": number, "error": str(error)})
    if limit is not None:
        records = records[-limit:]
    return records, errors


def create_app(run_root="runs", frontend_dir=None, cors_origins=None,
               project_root=None, data_root=None):
    (FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect,
     CORSMiddleware, FileResponse, StaticFiles) = _dashboard_dependencies()
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="KernelLeaf AutoDrive Dashboard", version="1.0")
    origins = cors_origins or ["http://localhost:5173", "http://127.0.0.1:5173"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.run_root = root
    app.state.training = TrainingProcessController(root)
    project_root_path = Path(
        project_root or Path(__file__).resolve().parents[2]
    ).resolve()
    data_root_path = Path(
        data_root or project_root_path / "data" / "DonkeyCar"
    ).resolve()

    def training_config(value):
        value = value if isinstance(value, dict) else {}

        def integer(name, default, minimum, maximum):
            try:
                result = int(value.get(name, default))
            except (TypeError, ValueError) as error:
                raise HTTPException(status_code=422, detail=f"{name} 必须是整数") from error
            if not minimum <= result <= maximum:
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} 必须位于 {minimum} 到 {maximum} 之间",
                )
            return result

        def number(name, default, minimum, *, positive=False):
            try:
                result = float(value.get(name, default))
            except (TypeError, ValueError) as error:
                raise HTTPException(status_code=422, detail=f"{name} 必须是数字") from error
            if not math.isfinite(result) or result < minimum or (positive and result == 0):
                qualifier = "大于" if positive else "不小于"
                raise HTTPException(status_code=422, detail=f"{name} 必须{qualifier} {minimum}")
            return result

        raw_manifest = str(value.get("manifest", "data/DonkeyCar/manifest.jsonl"))
        manifest = Path(raw_manifest)
        manifest = ((project_root_path / manifest).resolve()
                    if not manifest.is_absolute() else manifest.resolve())
        try:
            manifest.relative_to(data_root_path)
        except ValueError as error:
            raise HTTPException(
                status_code=422, detail="manifest 必须位于 data/DonkeyCar 目录内"
            ) from error
        if not manifest.is_file():
            raise HTTPException(status_code=422, detail=f"manifest 不存在：{manifest}")

        if "maps" in value:
            raise HTTPException(
                status_code=422, detail="不再支持 maps；请通过 map 指定一个地图"
            )
        raw_map = value.get("map", "")
        if not isinstance(raw_map, str) or not raw_map.strip():
            raise HTTPException(status_code=422, detail="map 必须指定一个地图")
        map_name = raw_map.strip()
        if len(map_name) > 128 or "," in map_name:
            raise HTTPException(status_code=422, detail="map 只能包含一个地图名称")
        available_maps = set()
        line_number = 0
        try:
            with manifest.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if isinstance(record, dict) and record.get("map_name"):
                        available_maps.add(str(record["map_name"]))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(
                status_code=422,
                detail=f"manifest 无法读取或包含损坏记录（第 {line_number} 行）：{error}",
            ) from error
        if map_name not in available_maps:
            raise HTTPException(
                status_code=422,
                detail=f"未知地图 {map_name!r}；可用地图：{sorted(available_maps)}",
            )

        device = str(value.get("device", "cpu")).lower()
        device = "cuda" if device == "gpu" else device
        if device == "auto":
            device = "cpu"
        if device not in {"cpu", "cuda"}:
            raise HTTPException(status_code=422, detail="device 必须是 cpu 或 cuda")
        return {
            "manifest": str(manifest), "map": map_name,
            "batch_size": integer("batch_size", 32, 1, 4096),
            "epochs": integer("epochs", 10, 1, 10000),
            "lr": number("lr", 0.001, 0, positive=True),
            "weight_decay": number("weight_decay", 0.0001, 0),
            "grad_clip_norm": number("grad_clip_norm", 5.0, 0),
            "device": device,
        }

    def unique_dashboard_run_id():
        base = default_run_id("dashboard")
        candidate = base
        suffix = 2
        while (root / candidate).exists():
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def training_command(config, run_id):
        command = [
            sys.executable, "-m", "apps.autodrive", "train",
            "--manifest", config["manifest"],
            "--device", config["device"],
            "--epochs", str(config["epochs"]),
            "--batch-size", str(config["batch_size"]),
            "--lr", str(config["lr"]),
            "--weight-decay", str(config["weight_decay"]),
            "--grad-clip-norm", str(config["grad_clip_norm"]),
            "--num-workers", "0",
            "--runs-root", str(root),
            "--run-name", run_id,
        ]
        command.extend(["--map", config["map"]])
        return command

    def available_runs():
        directories = []
        for directory in root.iterdir():
            if not directory.is_dir() or not directory.resolve().is_relative_to(root):
                continue
            try:
                validate_run_id(directory.name)
            except ValueError:
                continue
            metadata = optional_json(directory, "run.json")
            if isinstance(metadata["data"], dict):
                directories.append((directory, metadata["data"]))
        return sorted(
            directories,
            key=lambda item: item[1].get("updated_at", ""),
            reverse=True,
        )

    def current_run():
        choices = available_runs()
        return choices[0] if choices else (None, {})

    def current_bundle(limit=None):
        directory, metadata = current_run()
        if directory is None:
            return None, {}, [], None
        records, _ = _read_metrics(directory / "metrics.jsonl", limit)
        latest_record = records[-1] if records else None
        return directory, metadata, records, latest_record

    def timestamp_seconds(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value).timestamp()
        except (TypeError, ValueError):
            return None

    def system_snapshot(record=None):
        record = record or {}
        cpu_memory_mb = record.get("cpu_memory_mb") or 0
        total_memory_gb = used_memory_gb = memory_percent = 0
        disk_total_gb = disk_used_gb = disk_percent = 0
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(root.anchor)
            total_memory_gb = memory.total / 1024 ** 3
            used_memory_gb = cpu_memory_mb / 1024
            memory_percent = used_memory_gb / total_memory_gb * 100
            disk_total_gb = disk.total / 1024 ** 3
            disk_used_gb = disk.used / 1024 ** 3
            disk_percent = disk.percent
        except (ImportError, OSError, ZeroDivisionError):
            pass
        gpu_used = record.get("gpu_memory_mb") or 0
        gpu_total = record.get("gpu_memory_total_mb") or 0
        gpus = []
        if gpu_total:
            gpus.append({
                "index": 0, "name": "CUDA GPU", "util_percent": 0,
                "mem_used_mb": gpu_used, "mem_total_mb": gpu_total,
                "mem_percent": gpu_used / gpu_total * 100, "temp_c": 0,
            })
        return {
            "cpu_percent": record.get("cpu_percent") or 0,
            "memory_percent": memory_percent,
            "memory_used_gb": used_memory_gb,
            "memory_total_gb": total_memory_gb,
            "gpus": gpus,
            "disk_percent": disk_percent,
            "disk_used_gb": disk_used_gb,
            "disk_total_gb": disk_total_gb,
        }

    def legacy_status(metadata, record):
        config = metadata.get("config", {})
        state = metadata.get("status", "idle")
        return {
            "running": state == "running",
            "paused": False,
            "epoch": record.get("epoch", 0) if record else 0,
            "total_epochs": metadata.get("total_epochs", config.get("epochs", 0)),
            "steer_loss": record.get("steer_loss") if record else None,
            "throttle_loss": record.get("throttle_loss") if record else None,
            "best_steer_loss": record.get("steer_loss") if record else None,
            "best_throttle_loss": record.get("throttle_loss") if record else None,
            "lr": record.get("lr", config.get("lr", 0.001)) if record else config.get("lr", 0.001),
            "message": state,
            "started_at": timestamp_seconds(metadata.get("created_at")),
            "system": system_snapshot(record),
            "run_id": metadata.get("run_id"),
        }

    def run_directory(run_id):
        try:
            name = validate_run_id(run_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        candidate = (root / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise HTTPException(status_code=400, detail="run path escapes run root") from error
        if not candidate.is_dir():
            raise HTTPException(status_code=404, detail="run not found")
        return candidate

    def optional_json(directory, filename):
        path = directory / filename
        if not path.is_file():
            return {"data": None, "error": f"{filename} is missing"}
        try:
            return {"data": _read_json(path), "error": None}
        except (OSError, json.JSONDecodeError) as error:
            return {"data": None, "error": f"invalid {filename}: {error}"}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "run_root": str(root)}

    @app.get("/api/runs")
    def runs():
        items = []
        for directory in sorted(root.iterdir(), reverse=True):
            if not directory.is_dir() or not directory.resolve().is_relative_to(root):
                continue
            try:
                validate_run_id(directory.name)
            except ValueError:
                continue
            value = optional_json(directory, "run.json")
            metadata = value["data"] if isinstance(value["data"], dict) else {}
            items.append({
                "run_id": directory.name,
                "status": metadata.get("status", "invalid"),
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "total_epochs": metadata.get("total_epochs"),
                "config": metadata.get("config", {}),
                "error": value["error"],
            })
        return {"runs": items}

    # Compatibility facade for the original Paddle dashboard.  These routes
    # only read the newest KernelLeaf JSONL run; they never launch or mutate a
    # training process.
    @app.get("/api/train/status")
    def legacy_train_status():
        _, metadata, _, record = current_bundle(1)
        result = legacy_status(metadata, record)
        controlled = app.state.training.snapshot()
        if controlled["run_id"]:
            if record and record.get("run_id") != controlled["run_id"]:
                record = None
                result = legacy_status({}, None)
            result.update(
                running=controlled["running"], paused=controlled["paused"],
                run_id=controlled["run_id"], message=controlled["message"],
                console_log=controlled.get("console_log"),
                total_epochs=controlled["config"].get("epochs", result["total_epochs"]),
            )
        return result

    @app.get("/api/train/history")
    def legacy_train_history():
        _, _, records, _ = current_bundle(10000)
        best_steer = best_throttle = math.inf
        previous_step = 0
        epochs = []
        for record in records:
            steer = record.get("steer_loss")
            throttle = record.get("throttle_loss")
            is_best_steer = steer is not None and steer < best_steer
            is_best_throttle = throttle is not None and throttle < best_throttle
            if is_best_steer:
                best_steer = steer
            if is_best_throttle:
                best_throttle = throttle
            step = record.get("step", previous_step)
            epochs.append({
                "epoch": record["epoch"], "lr": record.get("lr") or 0,
                "train_loss": record.get("train_loss"),
                "val_loss": record.get("val_loss"),
                "accuracy": record.get("accuracy"),
                "steer_loss": steer, "throttle_loss": throttle,
                "best_steer_loss": None if math.isinf(best_steer) else best_steer,
                "best_throttle_loss": None if math.isinf(best_throttle) else best_throttle,
                "epoch_time": record.get("epoch_seconds") or 0,
                "n_batches": max(0, step - previous_step),
                "is_best_steer": is_best_steer,
                "is_best_throttle": is_best_throttle,
            })
            previous_step = step
        return {"epochs": epochs}

    @app.get("/api/train/config")
    def legacy_train_config():
        controlled = app.state.training.snapshot()
        if controlled["config"]:
            return controlled["config"]
        _, metadata = current_run()
        config = metadata.get("config", {})
        return {
            "manifest": config.get("manifest", "data/DonkeyCar/manifest.jsonl"),
            "map": config.get("map", ""),
            "batch_size": config.get("batch_size", 32),
            "epochs": config.get("epochs", metadata.get("total_epochs", 10)),
            "lr": config.get("lr", 0.001),
            "weight_decay": config.get("weight_decay", 0.0001),
            "grad_clip_norm": config.get("grad_clip_norm", 0),
            "device": config.get("device", "cpu"),
        }

    @app.post("/api/train/start")
    def start_training(value: dict = None):
        config = training_config(value)
        run_id = unique_dashboard_run_id()
        try:
            app.state.training.start(
                training_command(config, run_id), run_id, config, project_root_path
            )
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "message": "KernelLeaf 训练已启动", "run_id": run_id,
            "config": config,
            "console_log": app.state.training.snapshot().get("console_log"),
        }

    @app.post("/api/train/pause")
    def pause_training():
        try:
            app.state.training.pause()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"message": "训练已暂停"}

    @app.post("/api/train/resume")
    def resume_training():
        try:
            app.state.training.resume()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"message": "训练已恢复"}

    @app.post("/api/train/stop")
    def stop_training():
        stopped = app.state.training.stop()
        return {"message": "停止信号已发送" if stopped else "当前没有正在运行的训练"}

    @app.get("/api/system/current")
    def legacy_system_current():
        _, _, _, record = current_bundle(1)
        return system_snapshot(record)

    @app.get("/api/system/history")
    def legacy_system_history(limit: int = Query(300, ge=1, le=10000)):
        _, _, records, _ = current_bundle(limit)
        return {"samples": [
            {**system_snapshot(record), "timestamp": timestamp_seconds(record.get("timestamp"))}
            for record in records
        ]}

    @app.get("/api/model/info")
    def legacy_model_info():
        _, metadata = current_run()
        config = metadata.get("config", {})
        base_channels = int(config.get("base_channels", 16))
        blocks = config.get("blocks", [1, 1, 1])
        total_parameters = None
        try:
            from .model import AutoDriveResNet
            model = AutoDriveResNet(base_channels=base_channels, blocks=blocks)
            total_parameters = sum(math.prod(parameter.shape) for parameter in model.parameters())
        except (ImportError, TypeError, ValueError):
            pass
        return {
            "architecture": "AutoDriveResNet",
            "base_channels": base_channels,
            "blocks": blocks,
            "shared_backbone": True,
            "steer_model": {"parameters": base_channels * 4 + 1},
            "throttle_model": {"parameters": base_channels * 4 + 1},
            "total_parameters": total_parameters,
        }

    @app.get("/api/model/checkpoints")
    def legacy_model_checkpoints():
        _, metadata = current_run()
        checkpoint = metadata.get("config", {}).get("checkpoint")
        if not checkpoint:
            return {"checkpoints": []}
        path = Path(checkpoint)
        return {"checkpoints": [{
            "name": path.name,
            "display_name": path.name,
            "size_mb": round(path.stat().st_size / 1024 ** 2, 2) if path.is_file() else 0,
            "download_url": None,
        }]}

    @app.get("/api/viz/list")
    def legacy_viz_list():
        directory, metadata = current_run()
        if directory is None:
            return {"images": [], "manifest": []}
        run_id = metadata.get("run_id", directory.name)
        values = gradcam_index(run_id)["images"]
        manifest = [{
            "index": index, "target": "throttle" if "throttle" in item["name"].lower() else "steer",
            "image_url": item["url"], "name": item["name"],
        } for index, item in enumerate(values, 1)]
        return {"images": values, "manifest": manifest}

    @app.get("/api/eval/validation")
    def legacy_validation():
        directory, _, _, record = current_bundle(1)
        if directory is None:
            raise HTTPException(status_code=404, detail="no training run is available")
        value = optional_json(directory, "validation.json")
        if value["data"] is None:
            raise HTTPException(status_code=404, detail=value["error"])
        result = value["data"]
        data_summary = optional_json(directory, "data_summary.json")["data"]
        data_summary = data_summary if isinstance(data_summary, dict) else {}
        return {
            "steer_mse": result.get("steer_loss"),
            "throttle_mse": result.get("throttle_loss"),
            "steer_mae": result.get("steer_mae"),
            "throttle_mae": result.get("throttle_mae"),
            "time_sec": record.get("epoch_seconds", 0) if record else 0,
            "n_samples": data_summary.get("splits", {}).get("val", 0),
            "time_per_sample_ms": (
                (record.get("epoch_seconds", 0) * 1000 / data_summary.get("splits", {}).get("val", 1))
                if record and data_summary.get("splits", {}).get("val", 0) else None
            ),
            "epoch": result.get("epoch"),
        }

    @app.get("/api/data/info")
    def legacy_data_info():
        directory, _ = current_run()
        summary = optional_json(directory, "data_summary.json")["data"] if directory else None
        summary = summary if isinstance(summary, dict) else {}
        splits = summary.get("splits", {})
        return {
            "train": {"count": splits.get("train", 0), "list_file": None},
            "val": {"count": splits.get("val", 0), "list_file": None},
            "summary": summary,
        }

    @app.get("/api/data/files")
    def legacy_data_files(limit: int = Query(200, ge=1, le=1000)):
        del limit
        return {"files": [], "total": 0, "message": "V11 does not expose source image paths"}

    @app.get("/api/runs/{run_id}")
    def run_info(run_id: str):
        directory = run_directory(run_id)
        value = optional_json(directory, "run.json")
        if value["data"] is None:
            raise HTTPException(status_code=422, detail=value["error"])
        return value["data"]

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str, limit: int = Query(500, ge=1, le=10000)):
        records, errors = _read_metrics(run_directory(run_id) / "metrics.jsonl", limit)
        return {"metrics": records, "errors": errors}

    @app.get("/api/runs/{run_id}/latest")
    def latest(run_id: str):
        directory = run_directory(run_id)
        value = optional_json(directory, "latest.json")
        if value["data"] is None:
            records, errors = _read_metrics(directory / "metrics.jsonl", 1)
            return {"latest": records[-1] if records else None,
                    "errors": [value["error"], *errors]}
        return {"latest": value["data"], "errors": []}

    @app.get("/api/runs/{run_id}/validation")
    def validation(run_id: str):
        value = optional_json(run_directory(run_id), "validation.json")
        return {"validation": value["data"], "error": value["error"]}

    @app.get("/api/runs/{run_id}/data-summary")
    def data_summary(run_id: str):
        value = optional_json(run_directory(run_id), "data_summary.json")
        return {"summary": value["data"], "error": value["error"]}

    @app.get("/api/runs/{run_id}/gradcam")
    def gradcam_index(run_id: str):
        directory = run_directory(run_id) / "gradcam"
        images = [] if not directory.is_dir() else [
            {
                "name": path.name,
                "url": f"/api/runs/{run_id}/gradcam/{path.name}",
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        return {"images": images}

    @app.get("/api/runs/{run_id}/gradcam/{filename}")
    def gradcam_image(run_id: str, filename: str):
        if Path(filename).name != filename:
            raise HTTPException(status_code=400, detail="invalid image name")
        directory = run_directory(run_id) / "gradcam"
        path = (directory / filename).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError as error:
            raise HTTPException(status_code=400, detail="image path escapes run") from error
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])

    @app.websocket("/ws/runs/{run_id}")
    async def run_updates(websocket: WebSocket, run_id: str):
        origin = websocket.headers.get("origin")
        host = websocket.headers.get("host")
        same_origin = {f"http://{host}", f"https://{host}"} if host else set()
        if origin and origin not in set(origins) | same_origin:
            await websocket.close(code=1008, reason="origin is not allowed")
            return
        await websocket.accept()
        try:
            directory = run_directory(run_id)
            signature = None
            while True:
                latest_path = directory / "latest.json"
                current = latest_path.stat().st_mtime_ns if latest_path.exists() else 0
                if current != signature:
                    signature = current
                    value = optional_json(directory, "latest.json")
                    await websocket.send_json({
                        "latest": value["data"], "error": value["error"]
                    })
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except HTTPException as error:
            await websocket.send_json({"latest": None, "error": error.detail})
            await websocket.close(code=1008)

    @app.websocket("/ws/train")
    async def legacy_train_updates(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        host = websocket.headers.get("host")
        same_origin = {f"http://{host}", f"https://{host}"} if host else set()
        if origin and origin not in set(origins) | same_origin:
            await websocket.close(code=1008, reason="origin is not allowed")
            return
        await websocket.accept()
        signature = None
        try:
            while True:
                directory, metadata = current_run()
                latest_path = directory / "latest.json" if directory else None
                current = latest_path.stat().st_mtime_ns if latest_path and latest_path.exists() else 0
                controlled = app.state.training.snapshot()
                current_signature = (
                    current, controlled["running"], controlled["paused"],
                    controlled["message"], controlled["run_id"],
                )
                if current_signature != signature:
                    signature = current_signature
                    await websocket.send_json(legacy_train_status())
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    frontend = Path(frontend_dir).resolve() if frontend_dir else (
        Path(__file__).resolve().parent / "dashboard" / "dist"
    )
    if frontend.is_dir():
        assets = frontend / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        def frontend_index():
            return FileResponse(frontend / "index.html")

        @app.get("/{frontend_path:path}", include_in_schema=False)
        def frontend_route(frontend_path: str):
            if frontend_path.startswith(("api/", "ws/")):
                raise HTTPException(status_code=404, detail="not found")
            return FileResponse(frontend / "index.html")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cors-origin", action="append", default=None,
        help="allowed frontend origin; repeat for multiple origins",
    )
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "dashboard support requires `pip install -e '.[dashboard]'`"
        ) from error
    uvicorn.run(
        create_app(args.runs_root, cors_origins=args.cors_origin),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
