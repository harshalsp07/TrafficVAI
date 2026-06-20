"""Camera Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class CalibrationPoint(BaseModel):
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float


class CalibrationData(BaseModel):
    points: List[CalibrationPoint] = []
    homography_matrix: Optional[List[List[float]]] = None
    calibrated_at: Optional[str] = None


class ZoneConfig(BaseModel):
    restricted_zones: List[List[List[float]]] = Field(default_factory=list, description="List of polygon coordinates for restricted parking zones")
    lane_directions: List[dict] = Field(default_factory=list, description="Lane direction vectors")
    stop_lines: List[dict] = Field(default_factory=list, description="Stop line coordinates")
    speed_limits: dict = Field(default_factory=lambda: {"default": 60}, description="Speed limits per lane")


class CameraCreate(BaseModel):
    name: str
    rtsp_url: str = ""
    location_lat: float = 0.0
    location_lon: float = 0.0
    description: str = ""
    zone_config: Optional[ZoneConfig] = None


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    description: Optional[str] = None
    status: Optional[str] = None
    zone_config: Optional[ZoneConfig] = None


class CameraResponse(BaseModel):
    id: int
    name: str
    rtsp_url: str
    location_lat: float
    location_lon: float
    description: str
    status: str
    zone_config: Optional[dict] = None
    calibration_data: Optional[dict] = None
    created_at: str
