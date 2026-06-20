"""Training job management service."""
import os
import sys
import json
import signal
import subprocess
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingService:
    """Manages training job lifecycles."""

    _active_processes: Dict[str, subprocess.Popen] = {}

    def __init__(self, weights_dir: str = "./weights", training_dir: str = "./training"):
        self.weights_dir = Path(weights_dir)
        self.training_dir = Path(training_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def start_training(self, config: Dict[str, Any], run_id: str) -> bool:
        """Launch a training subprocess."""
        script_path = self.training_dir / "scripts" / "train.py"
        if not script_path.exists():
            logger.error(f"Training script not found: {script_path}")
            return False

        cmd = [
            sys.executable, str(script_path),
            "--config", config.get("config_path", "configs/vehicle_detection.yaml"),
            "--epochs", str(config.get("epochs", 10)),
            "--batch", str(config.get("batch_size", 16)),
            "--device", config.get("device", "cpu"),
            "--run-id", run_id,
        ]

        if config.get("model_name"):
            cmd.extend(["--model", config["model_name"]])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.training_dir),
            )
            self._active_processes[run_id] = process
            logger.info(f"Training started: run_id={run_id}, pid={process.pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return False

    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Get training job status."""
        process = self._active_processes.get(run_id)
        if not process:
            return {"run_id": run_id, "status": "not_found"}
        poll = process.poll()
        if poll is None:
            return {"run_id": run_id, "status": "running", "pid": process.pid}
        return {
            "run_id": run_id,
            "status": "completed" if poll == 0 else "failed",
            "return_code": poll,
        }

    def stop_training(self, run_id: str) -> bool:
        """Stop a running training job."""
        process = self._active_processes.get(run_id)
        if not process:
            return False
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        self._active_processes.pop(run_id, None)
        logger.info(f"Training stopped: run_id={run_id}")
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        """List available model weight files."""
        models = []
        for ext in ("*.pt", "*.onnx", "*.engine"):
            for p in self.weights_dir.glob(ext):
                stat = p.stat()
                models.append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_active": False,
                })
        return models

    def deploy_model(self, model_path: str) -> bool:
        """Set a model as the active inference model."""
        path = Path(model_path)
        if not path.exists():
            path = self.weights_dir / model_path
        if path.exists():
            os.environ["MODEL_PATH"] = str(path)
            logger.info(f"Model deployed: {path}")
            return True
        logger.error(f"Model not found: {model_path}")
        return False
