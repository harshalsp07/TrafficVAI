"""Pydantic schemas for traffic violation endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Response models ─────────────────────────────────────────────────────

class ViolationOut(BaseModel):
    """Single violation record returned by the API."""

    id: int
    violation_id: str
    violation_time: str
    camera_id: int
    track_id: int | None = None
    vehicle_class: str | None = None
    license_plate: str | None = None
    plate_confidence: float | None = None
    violation_type: str
    confidence: float
    evidence_image_path: str | None = None
    evidence_crop_path: str | None = None
    sha256_hash: str | None = None
    status: str = "pending"
    vehicle_hash: str | None = None
    created_at: str | None = None


class ViolationListOut(BaseModel):
    """Paginated list of violations."""

    items: list[ViolationOut]
    total: int
    limit: int
    offset: int


class ViolationStatsOut(BaseModel):
    """Summary statistics for the violation dashboard."""

    today_count: int
    by_type: list[dict[str, Any]]
    by_camera: list[dict[str, Any]]
    by_status: dict[str, int]


class CertificateOut(BaseModel):
    """BSA Section 63 certificate text for a violation."""

    violation_id: str
    certificate_text: str
    generated_at: str


# ── Request models ──────────────────────────────────────────────────────

class ViolationStatusUpdate(BaseModel):
    """Payload for updating a violation's review status."""

    status: str = Field(
        ...,
        pattern=r"^(confirmed|dismissed)$",
        description="New status – must be 'confirmed' or 'dismissed'.",
    )
