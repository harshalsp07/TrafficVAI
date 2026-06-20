"""Streaming endpoints for SSE events and live video feeds."""
import os
import time
import json
import cv2
import logging
import asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.db import repository as repo
from app.services.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Global Inference Engine shared for streaming
engine = InferenceEngine()


# ── SSE Generators ──────────────────────────────────────────────────────

async def _violation_event_generator(request: Request):
    """Generate SSE events for new violations."""
    last_id = 0
    while True:
        if await request.is_disconnected():
            break
        try:
            db = request.app.state.db
            new_violations, _ = await repo.list_violations(db, limit=5, offset=0)
            if new_violations:
                for v in new_violations:
                    vid = v.get("id", 0)
                    if vid > last_id:
                        last_id = vid
                        yield {
                            "event": "violation",
                            "data": json.dumps(v, default=str),
                        }
        except Exception:
            pass
        await asyncio.sleep(2)


@router.get("/stream/violations")
async def stream_violations(request: Request):
    """SSE endpoint for live violation feed."""
    try:
        from sse_starlette.sse import EventSourceResponse
        return EventSourceResponse(_violation_event_generator(request))
    except ImportError:
        return {"error": "sse-starlette not installed"}


async def _training_event_generator(request: Request, run_id: str):
    """Generate SSE events for training progress."""
    while True:
        if await request.is_disconnected():
            break
        try:
            db = request.app.state.db
            run = await repo.get_training_run(db, run_id)
            if run:
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "run_id": run_id,
                        "status": run.get("status"),
                        "current_epoch": run.get("current_epoch", 0),
                        "epochs": run.get("epochs", 0),
                        "loss": run.get("loss", 0),
                        "best_map": run.get("best_map", 0),
                    }),
                }
                if run.get("status") in ("completed", "failed", "stopped"):
                    break
        except Exception:
            pass
        await asyncio.sleep(3)


@router.get("/stream/training/{run_id}")
async def stream_training(request: Request, run_id: str):
    """SSE endpoint for training progress."""
    try:
        from sse_starlette.sse import EventSourceResponse
        return EventSourceResponse(_training_event_generator(request, run_id))
    except ImportError:
        return {"error": "sse-starlette not installed"}


# ── MJPEG Video Stream Generator ──────────────────────────────────────────

async def _camera_stream_generator(camera_id: int):
    """Acquire camera frame, run model inference, and stream raw MJPEG bytes with resilience."""
    from app.config import settings
    import numpy as np

    # Lazy load the model on first request if not loaded
    if not engine.is_loaded:
        model_p = settings.MODEL_PATH
        if not os.path.exists(model_p) and os.path.exists("./runs/detect/idd_lite_run/weights/best.pt"):
            model_p = "./runs/detect/idd_lite_run/weights/best.pt"
        engine.load_model(model_p, device=settings.DEVICE)

    # Camera mapping: Cam-01 & Cam-03 use Webcam 0. Others use loop video or mock
    if camera_id in (1, 3):
        source = 0
    else:
        # High quality city traffic video loop
        source = "https://assets.mixkit.co/videos/preview/mixkit-traffic-at-a-busy-intersection-in-the-city-4467-large.mp4"

    reconnect_attempts = getattr(settings, "STREAM_RECONNECT_ATTEMPTS", 10)
    base_delay = getattr(settings, "STREAM_RECONNECT_BASE_DELAY", 1.0)
    
    cap = cv2.VideoCapture(source)
    
    # Target rates
    processing_fps = getattr(settings, "PROCESSING_FPS", 1.0)
    display_fps = getattr(settings, "DISPLAY_FPS", 15.0)
    
    processing_interval = 1.0 / max(0.1, processing_fps)
    display_interval = 1.0 / max(1.0, display_fps)
    
    last_process_time = 0.0
    last_display_time = 0.0
    detections = []
    
    try:
        attempt = 0
        while True:
            # Reconnection logic
            if cap is None or not cap.isOpened():
                if attempt >= reconnect_attempts:
                    logger.error(f"Failed to reconnect to stream after {reconnect_attempts} attempts.")
                    # Show offline frame
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Stream Offline - Reconnect Failed", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    _, jpeg = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    await asyncio.sleep(1.0)
                    continue
                
                attempt += 1
                delay = min(30.0, base_delay * (2 ** attempt))
                logger.info(f"Stream disconnected. Reconnecting in {delay:.1f}s (Attempt {attempt}/{reconnect_attempts})...")
                
                # Yield reconnecting overlay frame
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Reconnecting... (Attempt {attempt}/{reconnect_attempts})", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                await asyncio.sleep(delay)
                if cap:
                    cap.release()
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    logger.info("Successfully reconnected to stream.")
                    attempt = 0
                continue

            # Read frame
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and source.endswith(".mp4"):
                    # Loop local/mixkit files
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # RTSP/Webcam disconnect
                    logger.warning("Failed to read frame. Triggering reconnect.")
                    cap.release()
                    cap = None
                    continue

            now = time.time()
            
            # FPS Throttling for display
            if now - last_display_time < display_interval:
                # Read frames from cap to drain buffer but don't process/yield
                await asyncio.sleep(0.005)
                continue
                
            last_display_time = now

            # FPS Throttling for processing (inference)
            if now - last_process_time >= processing_interval:
                last_process_time = now
                if engine.is_loaded and engine.model:
                    try:
                        # Run model detection
                        detections = engine.detect(frame)
                    except Exception as e:
                        logger.error(f"Inference error in stream: {e}")
                        detections = []

            # Draw bounding boxes (either new or last cached ones)
            if detections:
                for det in detections:
                    bbox = [int(v) for v in det.bbox]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (16, 185, 129), 2)
                    cv2.putText(
                        frame, 
                        f"{det.class_name} {int(det.confidence * 100)}%", 
                        (bbox[0], bbox[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (16, 185, 129), 
                        1
                    )
            
            # Draw watermark or signal state if helpful
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            cv2.putText(frame, f"CAM_{camera_id:02d} | Live: {timestamp_str}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Encode as JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Yield back control to event loop
            await asyncio.sleep(0.01)

    except Exception as e:
        logger.error(f"Critical error in MJPEG stream generator: {e}")
    finally:
        if cap:
            cap.release()


@router.get("/stream/camera/{camera_id}")
async def stream_camera(camera_id: int):
    """Retrieve live camera stream with active YOLO detections."""
    return StreamingResponse(
        _camera_stream_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
