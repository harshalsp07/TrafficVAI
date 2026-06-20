"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the ITS backend.

    Values can be overridden via environment variables or a ``.env`` file
    located in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/traffic.db"

    # ── Model / Inference ───────────────────────────────────────────────
    MODEL_PATH: str = "./weights/yolov12n.pt"
    DEVICE: str = "cpu"
    CONFIDENCE_THRESHOLD: float = 0.5
    NMS_THRESHOLD: float = 0.45
    MAX_STREAMS: int = 4
    FRAME_SKIP: int = 1

    # ── Roboflow Inference ──────────────────────────────────────────────
    ROBOFLOW_ENABLED: bool = False
    ROBOFLOW_API_KEY: str = ""
    ROBOFLOW_API_URL: str = "https://serverless.roboflow.com"
    ROBOFLOW_MODEL_ID: str = "idd-octso/3,pedestrian-yldth/4,motorcycle-helmet-pz9xs-qoh78/1,seatbelt-detection-lb1ec-pjbz0/1"

    # ── SAHI (Sliced Inference) ─────────────────────────────────────────
    SAHI_ENABLED: bool = False
    SAHI_SLICE_HEIGHT: int = 640
    SAHI_SLICE_WIDTH: int = 640
    SAHI_OVERLAP_RATIO: float = 0.2

    # ── Violation Thresholds ─────────────────────────────────────────────
    SPEED_LIMIT_KMH: float = 60.0
    HELMET_CONFIDENCE_THRESHOLD: float = 0.6
    PARKING_DURATION_SECONDS: int = 300
    WRONG_SIDE_COSINE_THRESHOLD: float = -0.5
    MIN_CONSECUTIVE_FRAMES: int = 5

    # ── ANPR ────────────────────────────────────────────────────────────
    ANPR_ENABLED: bool = True
    OCR_ENGINE: str = "easyocr"
    PLATE_REGEX: str = "^[A-Z]{2}\\s?\\d{1,2}\\s?[A-Z]{1,3}\\s?\\d{4}$"

    # ── Micro-Classifiers ───────────────────────────────────────────────
    SEATBELT_MODEL_PATH: str = "./weights/seatbelt_mobilenetv3.pt"
    TRAFFIC_LIGHT_MODEL_PATH: str = "./weights/traffic_light_resnet18.pt"

    # ── Storage ─────────────────────────────────────────────────────────
    EVIDENCE_DIR: str = "./evidence"
    EVIDENCE_RETENTION_HOURS: int = 72

    # ── Server ──────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "*"

    # ── Dedup Configuration ──────────────────────────────────────────────
    DEDUP_HASH_THRESHOLD: int = 10          # Perceptual hash hamming distance
    DEDUP_HASH_WINDOW_MINUTES: int = 5      # Hash comparison time window
    DEDUP_DB_WINDOW_MINUTES: int = 5        # DB-level dedup time window

    # ── Stream Resilience ───────────────────────────────────────────────
    PROCESSING_FPS: float = 1.0             # Target FPS for violation detection
    DISPLAY_FPS: float = 15.0               # Target FPS for WebSocket display
    STREAM_RECONNECT_ATTEMPTS: int = 10
    STREAM_RECONNECT_BASE_DELAY: float = 1.0

    # ── Vehicle Classification Constants ────────────────────────────────
    MOTORIZED_VEHICLES: list[str] = ["car", "truck", "bus", "motorcycle", "auto_rickshaw", "van"]
    FOUR_WHEELERS: list[str] = ["car", "truck", "bus", "van"]
    TWO_WHEELERS: list[str] = ["motorcycle", "bicycle"]

    @property
    def sqlite_path(self) -> str:
        """Return the raw SQLite file path extracted from the DSN."""
        # "sqlite+aiosqlite:///./data/traffic.db" → "./data/traffic.db"
        return self.DATABASE_URL.split("///", maxsplit=1)[-1]

    def ensure_directories(self) -> None:
        """Create data and evidence directories if they don't exist."""
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.EVIDENCE_DIR).mkdir(parents=True, exist_ok=True)


# Singleton – importable from anywhere as ``from app.config import settings``
settings = Settings()
