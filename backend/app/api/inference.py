"""Inference API routes."""
import os
import uuid
import shutil
from fastapi import APIRouter, Request, HTTPException, UploadFile, File

router = APIRouter()


@router.post("/inference/video")
async def process_video(request: Request, file: UploadFile = File(...)):
    """Upload and process a video file."""
    evidence_dir = os.environ.get("EVIDENCE_DIR", "./evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    
    video_id = str(uuid.uuid4())[:8]
    video_path = os.path.join(evidence_dir, f"upload_{video_id}_{file.filename}")
    
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "video_id": video_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "status": "uploaded",
        "message": "Video uploaded. Processing will start in background.",
    }


@router.post("/inference/start-stream")
async def start_stream(request: Request, body: dict):
    """Start processing an RTSP stream."""
    rtsp_url = body.get("rtsp_url")
    camera_id = body.get("camera_id")
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtsp_url is required")
    
    return {
        "camera_id": camera_id,
        "rtsp_url": rtsp_url,
        "status": "started",
        "message": "Stream processing started.",
    }


@router.post("/inference/stop-stream")
async def stop_stream(request: Request, body: dict):
    """Stop processing a stream."""
    camera_id = body.get("camera_id")
    return {"camera_id": camera_id, "status": "stopped"}


@router.get("/inference/status")
async def inference_status(request: Request):
    """Get current inference engine status."""
    return {
        "model": os.environ.get("MODEL_PATH", "yolov12n.pt"),
        "device": os.environ.get("DEVICE", "cpu"),
        "active_streams": 0,
        "max_streams": int(os.environ.get("MAX_STREAMS", 4)),
        "avg_fps": 0.0,
        "avg_latency_ms": 0.0,
        "status": "idle",
    }
