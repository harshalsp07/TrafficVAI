"""FastAPI application entry point for TrafficAI ITS."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.db import database
from app.api import violations, cameras, analytics, training, inference, stream, reports, settings as settings_api


from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    db = await database.connect()
    app.state.db = db
    
    # Bootstrap settings from DB
    try:
        from app.db import repository as repo
        db_settings = await repo.list_settings(db)
        if "roboflow_enabled" in db_settings:
            settings.ROBOFLOW_ENABLED = db_settings["roboflow_enabled"].lower() == "true"
        if "roboflow_api_key" in db_settings:
            settings.ROBOFLOW_API_KEY = db_settings["roboflow_api_key"]
        if "roboflow_model_id" in db_settings:
            settings.ROBOFLOW_MODEL_ID = db_settings["roboflow_model_id"]
        if "roboflow_api_url" in db_settings:
            settings.ROBOFLOW_API_URL = db_settings["roboflow_api_url"]
        if "processing_fps" in db_settings:
            settings.PROCESSING_FPS = float(db_settings["processing_fps"])
        if "speed_limit" in db_settings:
            settings.SPEED_LIMIT_KMH = float(db_settings["speed_limit"])
        if "helmet_conf" in db_settings:
            settings.HELMET_CONFIDENCE_THRESHOLD = float(db_settings["helmet_conf"])
        if "parking_duration" in db_settings:
            settings.PARKING_DURATION_SECONDS = int(db_settings["parking_duration"])
        if "wrong_side_cosine" in db_settings:
            settings.WRONG_SIDE_COSINE_THRESHOLD = float(db_settings["wrong_side_cosine"])
        if "min_frames" in db_settings:
            settings.MIN_CONSECUTIVE_FRAMES = int(db_settings["min_frames"])
        if "ocr_engine" in db_settings:
            settings.OCR_ENGINE = db_settings["ocr_engine"]
            
        import logging
        logging.getLogger(__name__).info("Dynamic system settings loaded from database.")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to bootstrap settings: {e}")
        
    yield
    await database.disconnect()


app = FastAPI(
    title="TrafficAI - Intelligent Transportation System",
    description="Automated multi-violation traffic enforcement API",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(violations.router, prefix="/api", tags=["Violations"])
app.include_router(cameras.router, prefix="/api", tags=["Cameras"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])
app.include_router(training.router, prefix="/api", tags=["Training"])
app.include_router(inference.router, prefix="/api", tags=["Inference"])
app.include_router(stream.router, prefix="/api", tags=["Streaming"])
app.include_router(reports.router, prefix="/api", tags=["Reports"])
app.include_router(settings_api.router, prefix="/api", tags=["Settings"])

os.makedirs("evidence", exist_ok=True)
try:
    app.mount("/evidence", StaticFiles(directory="evidence"), name="evidence")
except Exception:
    pass


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "TrafficAI ITS"}
