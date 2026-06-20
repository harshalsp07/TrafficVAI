#!/usr/bin/env python3
"""Unified YOLOv12 training script for the ITS traffic model."""
import argparse
import json
import yaml
import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def setup_logging(run_dir: Path) -> logging.Logger:
    """Configure file and console logging."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("its_training")
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(run_dir / "training.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_config(config_path: str) -> dict:
    """Load and validate YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        print(json.dumps({"type": "error", "message": f"Config not found: {config_path}"}))
        sys.exit(1)
    with open(path) as f:
        config = yaml.safe_load(f)
    return config or {}


def make_progress_callback(total_epochs: int):
    """Create an epoch-end callback that outputs JSON progress."""
    def on_train_epoch_end(trainer):
        try:
            metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
            loss_val = getattr(trainer, 'loss', 0)
            if hasattr(loss_val, 'detach'):
                loss_val = loss_val.detach().item()
            else:
                loss_val = float(loss_val)
            progress = {
                "type": "training_progress",
                "epoch": trainer.epoch + 1,
                "total_epochs": total_epochs,
                "loss": loss_val,
                "metrics": {
                    "map50": float(metrics.get("metrics/mAP50(B)", 0)),
                    "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0)),
                    "precision": float(metrics.get("metrics/precision(B)", 0)),
                    "recall": float(metrics.get("metrics/recall(B)", 0)),
                },
                "lr": float(list(trainer.lr.values())[0]) if hasattr(trainer, 'lr') and trainer.lr else 0.0,
                "timestamp": datetime.now().isoformat(),
            }
            print(json.dumps(progress), flush=True)
        except Exception as e:
            print(json.dumps({"type": "warning", "message": str(e)}), flush=True)
    return on_train_epoch_end


def train(config: dict, overrides: dict, logger: logging.Logger):
    """Run model training."""
    model_path = overrides.get("model", config.get("model", "yolov12n.pt"))
    epochs = overrides.get("epochs", config.get("epochs", 100))
    batch = overrides.get("batch", config.get("batch", 16))
    device = overrides.get("device", config.get("device", ""))
    imgsz = config.get("imgsz", 640)
    data = config.get("data")
    
    logger.info(f"Training config: model={model_path}, epochs={epochs}, batch={batch}, device={device}")
    
    model = YOLO(model_path)
    model.add_callback("on_train_epoch_end", make_progress_callback(epochs))
    
    train_params = {
        "data": data,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device if device else None,
        "workers": config.get("workers", 4),
        "patience": config.get("patience", 20),
        "save_period": config.get("save_period", 10),
        "optimizer": config.get("optimizer", "AdamW"),
        "lr0": config.get("lr0", 0.001),
        "lrf": config.get("lrf", 0.01),
        "momentum": config.get("momentum", 0.937),
        "weight_decay": config.get("weight_decay", 0.0005),
        "warmup_epochs": config.get("warmup_epochs", 3.0),
        "mosaic": config.get("mosaic", 1.0),
        "mixup": config.get("mixup", 0.15),
        "flipud": config.get("flipud", 0.0),
        "fliplr": config.get("fliplr", 0.5),
        "hsv_h": config.get("hsv_h", 0.015),
        "hsv_s": config.get("hsv_s", 0.7),
        "hsv_v": config.get("hsv_v", 0.4),
        "translate": config.get("translate", 0.1),
        "scale": config.get("scale", 0.5),
        "box": config.get("box", 7.5),
        "cls": config.get("cls", 0.5),
        "dfl": config.get("dfl", 1.5),
    }
    train_params = {k: v for k, v in train_params.items() if v is not None}
    
    if overrides.get("resume"):
        train_params["resume"] = True
    
    results = model.train(**train_params)
    
    summary = {
        "type": "training_complete",
        "model": model_path,
        "epochs": epochs,
        "results_dir": str(getattr(results, 'save_dir', 'N/A')),
        "timestamp": datetime.now().isoformat(),
    }
    print(json.dumps(summary), flush=True)
    logger.info(f"Training complete: {summary}")
    return results


def main():
    parser = argparse.ArgumentParser(description="ITS Traffic Model Training")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--device", type=str, help="Override device (cpu, 0, cuda:0)")
    parser.add_argument("--model", type=str, help="Override model path or name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--run-id", type=str, default="", help="Training run ID for tracking")
    parser.add_argument("--output-dir", type=str, default="./runs", help="Output directory")
    args = parser.parse_args()
    
    if not HAS_ULTRALYTICS:
        print(json.dumps({
            "type": "error",
            "message": "ultralytics not installed. Install with: pip install ultralytics"
        }))
        sys.exit(1)
    
    run_name = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    logger = setup_logging(run_dir)
    
    config = load_config(args.config)
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.device is not None:
        overrides["device"] = args.device
    if args.model is not None:
        overrides["model"] = args.model
    if args.resume:
        overrides["resume"] = True
    
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Overrides: {overrides}")
    
    try:
        train(config, overrides, logger)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
