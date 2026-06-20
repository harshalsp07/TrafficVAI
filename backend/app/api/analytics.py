"""Analytics API routes."""
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Request, Query
from app.db import repository as repo

router = APIRouter()


@router.get("/analytics/trends")
async def get_trends(request: Request, days: int = Query(7, ge=1, le=90)):
    """Get violation trends over time."""
    db = request.app.state.db
    date_from = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    trends = await repo.violation_trends(db, granularity="daily", date_from=date_from)
    return {"data": trends}


@router.get("/analytics/heatmap")
async def get_heatmap(request: Request):
    """Get violation locations for map heatmap."""
    db = request.app.state.db
    points = await repo.violation_heatmap(db)
    return {"data": points}


@router.get("/analytics/distribution")
async def get_distribution(request: Request, group_by: str = Query("violation_type")):
    """Get violation distribution by type or vehicle class."""
    db = request.app.state.db
    distribution = await repo.violation_distribution(db, group_by=group_by)
    return {"data": distribution}


@router.get("/analytics/speed")
async def get_speed_data(request: Request, camera_id: int = None):
    """Get speed distribution data."""
    db = request.app.state.db
    speed_data = await repo.speed_distribution(db, camera_id=camera_id)
    return {"data": speed_data}


@router.get("/analytics/density")
async def get_density(request: Request, camera_id: int = None):
    """Get traffic density by camera/hour."""
    db = request.app.state.db
    density = await repo.get_density(db, camera_id=camera_id)
    return {"data": density}


@router.get("/analytics/peak-hours")
async def get_peak_hours(request: Request):
    """Get peak violation hours matrix."""
    db = request.app.state.db
    peak_data = await repo.peak_hours(db)
    return {"data": peak_data}
