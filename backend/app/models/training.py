"""Training Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class TrainingConfig(BaseModel):
    model_name: str = "yolov12n.pt"
    dataset: str = "datasets/sample/dataset.yaml"
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.001
    device: str = "cpu"
    resume: bool = False
    config_path: Optional[str] = None


class TrainingProgress(BaseModel):
    epoch: int
    total_epochs: int
    loss: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    lr: float = 0.0
    eta: Optional[str] = None


class TrainingRun(BaseModel):
    id: Optional[int] = None
    run_id: str
    model_name: str
    dataset: str
    epochs: int
    batch_size: int
    status: str = "queued"
    current_epoch: int = 0
    best_map: float = 0.0
    loss: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    modified_at: str
    is_active: bool = False
