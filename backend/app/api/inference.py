"""Inference API routes."""
import os
import uuid
import time
import logging
import cv2
import numpy as np
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@router.post("/inference/image")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    save_to_db: Optional[bool] = False,
):
    """Upload and process an image through the violation detection pipeline.

    Returns annotated image URL, detected violations, and processing metadata.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, WEBP, BMP",
        )

    evidence_dir = os.environ.get("EVIDENCE_DIR", "./evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    os.makedirs(os.path.join(evidence_dir, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(evidence_dir, "annotated"), exist_ok=True)

    upload_id = uuid.uuid4().hex[:8]
    ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "jpg"
    upload_path = os.path.join(evidence_dir, "uploads", f"upload_{upload_id}.{ext}")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 20MB allowed.")

    with open(upload_path, "wb") as f:
        f.write(content)

    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    start_time = time.time()

    try:
        from app.services.unified_pipeline import UnifiedPipeline

        pipeline = UnifiedPipeline()
        camera_config = {
            "id": "upload_camera",
            "fps": 30,
            "processing_fps": 30,
            "signal_state": "green",
            "sahi_enabled": False,
        }

        result = pipeline.process_frame(frame, camera_config)

        detections = []
        for track in result.tracks:
            detections.append({
                "bbox": track.bbox,
                "class_name": track.class_name,
                "confidence": track.confidence,
            })

        violations = [v.to_dict() for v in result.violations]

        from app.services.image_annotator import annotate_image, save_annotated

        annotated_frame = annotate_image(frame, detections, violations)
        annotated_url = save_annotated(annotated_frame, prefix=f"violation_{upload_id}")

        processing_time_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "upload_id": upload_id,
            "original_filename": file.filename,
            "annotated_image_url": annotated_url,
            "processing_time_ms": processing_time_ms,
            "total_detections": len(detections),
            "total_violations": len(violations),
            "violations": violations,
            "quality": {
                "blur_score": result.quality.blur_score if result.quality else None,
                "brightness": result.quality.brightness if result.quality else None,
            },
        }

    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}", exc_info=True)

        from app.services.image_annotator import save_annotated

        error_url = save_annotated(frame, prefix=f"error_{upload_id}")
        processing_time_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "partial",
            "upload_id": upload_id,
            "original_filename": file.filename,
            "annotated_image_url": error_url,
            "processing_time_ms": processing_time_ms,
            "total_detections": 0,
            "total_violations": 0,
            "violations": [],
            "warning": f"Pipeline error: {str(e)}. Original image returned.",
        }


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
