"""System settings API router."""
import logging
from fastapi import APIRouter, Request, HTTPException
from app.config import settings
from app.db import repository as repo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/settings")
async def get_system_settings(request: Request):
    """Retrieve all configurable system settings."""
    db = request.app.state.db
    db_settings = await repo.list_settings(db)
    
    # Calculate capture_interval_seconds (seconds between images)
    # processing_fps = 1.0 / capture_interval_seconds
    processing_fps = float(db_settings.get("processing_fps", settings.PROCESSING_FPS))
    capture_interval = 1.0 / max(0.01, processing_fps)
    
    return {
        "roboflow_enabled": db_settings.get("roboflow_enabled", str(settings.ROBOFLOW_ENABLED)).lower() == "true",
        "roboflow_api_key": db_settings.get("roboflow_api_key", settings.ROBOFLOW_API_KEY),
        "roboflow_model_id": db_settings.get("roboflow_model_id", settings.ROBOFLOW_MODEL_ID),
        "roboflow_api_url": db_settings.get("roboflow_api_url", settings.ROBOFLOW_API_URL),
        "capture_interval_seconds": round(capture_interval, 2),
        
        # Thresholds
        "speed_limit": float(db_settings.get("speed_limit", settings.SPEED_LIMIT_KMH)),
        "helmet_conf": float(db_settings.get("helmet_conf", settings.HELMET_CONFIDENCE_THRESHOLD)),
        "parking_duration": int(db_settings.get("parking_duration", settings.PARKING_DURATION_SECONDS)),
        "wrong_side_cosine": float(db_settings.get("wrong_side_cosine", settings.WRONG_SIDE_COSINE_THRESHOLD)),
        "min_frames": int(db_settings.get("min_frames", settings.MIN_CONSECUTIVE_FRAMES)),
        "ocr_engine": db_settings.get("ocr_engine", settings.OCR_ENGINE),
        "region": db_settings.get("region", "IN"),
    }


@router.post("/settings")
async def update_system_settings(request: Request, body: dict):
    """Update system settings and persist to database & in-memory config."""
    db = request.app.state.db
    
    try:
        if "roboflow_enabled" in body:
            val = bool(body["roboflow_enabled"])
            await repo.set_setting(db, "roboflow_enabled", str(val))
            settings.ROBOFLOW_ENABLED = val
            
        if "roboflow_api_key" in body:
            val = str(body["roboflow_api_key"])
            await repo.set_setting(db, "roboflow_api_key", val)
            settings.ROBOFLOW_API_KEY = val
            
        if "roboflow_model_id" in body:
            val = str(body["roboflow_model_id"])
            await repo.set_setting(db, "roboflow_model_id", val)
            settings.ROBOFLOW_MODEL_ID = val
            
        if "roboflow_api_url" in body:
            val = str(body["roboflow_api_url"])
            await repo.set_setting(db, "roboflow_api_url", val)
            settings.ROBOFLOW_API_URL = val
            
        if "capture_interval_seconds" in body:
            interval = max(0.1, float(body["capture_interval_seconds"]))
            fps = 1.0 / interval
            await repo.set_setting(db, "processing_fps", str(fps))
            settings.PROCESSING_FPS = fps
            
        if "speed_limit" in body:
            val = float(body["speed_limit"])
            await repo.set_setting(db, "speed_limit", str(val))
            settings.SPEED_LIMIT_KMH = val
            
        if "helmet_conf" in body:
            val = float(body["helmet_conf"])
            await repo.set_setting(db, "helmet_conf", str(val))
            settings.HELMET_CONFIDENCE_THRESHOLD = val
            
        if "parking_duration" in body:
            val = int(body["parking_duration"])
            await repo.set_setting(db, "parking_duration", str(val))
            settings.PARKING_DURATION_SECONDS = val
            
        if "wrong_side_cosine" in body:
            val = float(body["wrong_side_cosine"])
            await repo.set_setting(db, "wrong_side_cosine", str(val))
            settings.WRONG_SIDE_COSINE_THRESHOLD = val
            
        if "min_frames" in body:
            val = int(body["min_frames"])
            await repo.set_setting(db, "min_frames", str(val))
            settings.MIN_CONSECUTIVE_FRAMES = val

        if "ocr_engine" in body:
            val = str(body["ocr_engine"])
            await repo.set_setting(db, "ocr_engine", val)
            settings.OCR_ENGINE = val
            
        # Re-initialize the model loader in stream or main services if needed
        # We trigger this by setting is_loaded = False on the stream engine
        try:
            from app.api.stream import engine as stream_engine
            stream_engine.is_loaded = False
            logger.info("Signaled stream engine to reload models on next frame.")
        except Exception as e:
            logger.warning(f"Could not reset stream engine load flag: {e}")
        
        return {"status": "success", "message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")
