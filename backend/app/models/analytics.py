"""Analytics Pydantic models."""
from pydantic import BaseModel
from typing import List, Optional


class TrendPoint(BaseModel):
    timestamp: str
    count: int
    violation_type: Optional[str] = None


class HeatmapPoint(BaseModel):
    lat: float
    lon: float
    intensity: int
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None


class DistributionItem(BaseModel):
    category: str
    count: int
    percentage: float = 0.0


class SpeedBucket(BaseModel):
    range_start: int
    range_end: int
    count: int


class DensityRecord(BaseModel):
    window_hour: str
    camera_id: str
    lane_id: int
    total_vehicles: int
    average_speed: float


class PeakHour(BaseModel):
    hour: int
    day_of_week: int
    count: int
