"""Camera API routes."""
from fastapi import APIRouter, Request, HTTPException
from app.models.camera import CameraCreate, CameraUpdate, CalibrationData
from app.db import repository as repo

router = APIRouter()


@router.get("/cameras")
async def list_cameras(request: Request):
    """List all cameras."""
    db = request.app.state.db
    cameras = await repo.list_cameras(db)
    return {"data": cameras}


@router.post("/cameras")
async def create_camera(request: Request, camera: CameraCreate):
    """Add a new camera."""
    db = request.app.state.db
    result = await repo.create_camera(
        db,
        name=camera.name,
        rtsp_url=camera.rtsp_url,
        location_lat=camera.location_lat,
        location_lon=camera.location_lon,
    )
    return {"id": result.get("id"), "message": "Camera created successfully"}


@router.get("/cameras/{camera_id}")
async def get_camera(request: Request, camera_id: int):
    """Get camera details."""
    db = request.app.state.db
    camera = await repo.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.put("/cameras/{camera_id}")
async def update_camera(request: Request, camera_id: int, camera: CameraUpdate):
    """Update camera configuration."""
    db = request.app.state.db
    update_data = {k: v for k, v in camera.model_dump().items() if v is not None}
    await repo.update_camera(db, camera_id, **update_data)
    return {"id": camera_id, "message": "Camera updated"}


@router.delete("/cameras/{camera_id}")
async def delete_camera(request: Request, camera_id: int):
    """Remove a camera."""
    db = request.app.state.db
    await repo.delete_camera(db, camera_id)
    return {"message": "Camera deleted"}


@router.post("/cameras/{camera_id}/calibrate")
async def calibrate_camera(request: Request, camera_id: int, data: CalibrationData):
    """Store calibration data for a camera."""
    db = request.app.state.db
    await repo.update_camera(db, camera_id, calibration_data=data.model_dump())
    return {"id": camera_id, "message": "Calibration data stored"}
