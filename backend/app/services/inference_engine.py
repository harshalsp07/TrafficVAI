"""Inference engine service for YOLOv12 object detection."""
import os
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


from app.config import settings

@dataclass
class Detection:
    """Single detection result."""
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    track_id: Optional[int] = None


class InferenceEngine:
    """YOLOv12 inference engine with SAHI and model fallback, or Roboflow Universe hosted API."""

    FALLBACK_CHAIN = ["yolov12", "yolov11", "yolov8"]

    def __init__(self):
        self.model = None
        self.model_path = ""
        self.device = "cpu"
        self.confidence_threshold = 0.5
        self.current_fps = 0.0
        self.frame_count = 0
        self.is_loaded = False
        self.sahi_model = None

        # Roboflow configuration
        self.roboflow_enabled = False
        self.roboflow_api_key = ""
        self.roboflow_api_url = ""
        self.roboflow_model_ids = []
        self.roboflow_client = None
        self.class_mapping = {
            # IDD model classes (vehicle-detection)
            "car": "car",
            "vehicle": "car",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motorcycle",
            "motorbike": "motorcycle",
            "bike": "motorcycle",
            "bicycle": "bicycle",
            "cycle": "bicycle",
            "auto_rickshaw": "auto_rickshaw",
            "rickshaw": "auto_rickshaw",
            "van": "van",
            "autorickshaw": "auto_rickshaw",
            # Pedestrian / rider model classes
            "pedestrian": "pedestrian",
            "person": "rider",
            "rider": "rider",
            # Helmet model classes
            "helmet": "helmet",
            "with-helmet": "helmet",
            "without-helmet": "no_helmet",
            "no-helmet": "no_helmet",
            "no_helmet": "no_helmet",
            # Seatbelt model classes
            "seatbelt": "seatbelt",
            "with-seatbelt": "seatbelt",
            "without-seatbelt": "no_seatbelt",
            "no-seatbelt": "no_seatbelt",
            "no_seatbelt": "no_seatbelt",
            # License plate
            "license_plate": "license_plate",
            "license-plate": "license_plate",
            "number-plate": "license_plate",
            "number_plate": "license_plate",
            "plate": "license_plate",
            # Traffic light
            "traffic_light": "traffic_light",
            "traffic light": "traffic_light",
            "signal": "traffic_light",
        }

    def load_model(self, path: str, device: str = "cpu") -> bool:
        """Load a YOLO model or initialize Roboflow config."""
        self.sahi_model = None
        
        # Check settings for Roboflow configuration first
        if getattr(settings, "ROBOFLOW_ENABLED", False):
            self.roboflow_enabled = True
            self.roboflow_api_key = getattr(settings, "ROBOFLOW_API_KEY", "")
            self.roboflow_api_url = getattr(settings, "ROBOFLOW_API_URL", "https://serverless.roboflow.com")
            model_ids_str = getattr(settings, "ROBOFLOW_MODEL_ID", "")
            self.roboflow_model_ids = [m.strip() for m in model_ids_str.split(",") if m.strip()]
            self.device = device
            self.confidence_threshold = getattr(settings, "CONFIDENCE_THRESHOLD", 0.5)
            self._init_roboflow_client()
            self.is_loaded = True
            self.model_path = f"roboflow://{model_ids_str}"
            logger.info(f"Roboflow API client initialized with models: {self.roboflow_model_ids}")
            return True

        # Check if path is a roboflow URI format directly
        if path and str(path).startswith("roboflow://"):
            self.roboflow_enabled = True
            self.roboflow_api_key = getattr(settings, "ROBOFLOW_API_KEY", "")
            self.roboflow_api_url = getattr(settings, "ROBOFLOW_API_URL", "https://serverless.roboflow.com")
            model_ids_str = str(path).replace("roboflow://", "")
            self.roboflow_model_ids = [m.strip() for m in model_ids_str.split(",") if m.strip()]
            self.device = device
            self.confidence_threshold = getattr(settings, "CONFIDENCE_THRESHOLD", 0.5)
            self._init_roboflow_client()
            self.is_loaded = True
            self.model_path = path
            logger.info(f"Roboflow API client initialized via path URI with models: {self.roboflow_model_ids}")
            return True

        # Fallback to local YOLO
        self.roboflow_enabled = False
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.model_path = path
            self.device = device
            self.is_loaded = True
            logger.info(f"Model loaded: {path} on {device}")
            return True
        except ImportError:
            logger.warning("ultralytics not installed. Running in mock mode.")
            self.model_path = path
            self.device = device
            self.is_loaded = False
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False

    def _init_roboflow_client(self):
        """Initialize the Roboflow inference-sdk client."""
        try:
            from inference_sdk import InferenceHTTPClient
            self.roboflow_client = InferenceHTTPClient(
                api_url=self.roboflow_api_url,
                api_key=self.roboflow_api_key,
            )
            logger.info(f"Roboflow InferenceHTTPClient initialized: url={self.roboflow_api_url}")
        except ImportError:
            logger.warning("inference-sdk not installed. Falling back to raw HTTP.")
            self.roboflow_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Roboflow client: {e}")
            self.roboflow_client = None

    def _detect_roboflow(self, frame) -> List[Detection]:
        """Run detection using Roboflow Inference SDK on one or more models."""
        if not self.roboflow_api_key:
            logger.warning("Roboflow API key not configured. Skipping detection.")
            return []

        import cv2

        try:
            # Save frame to temp file for inference-sdk (it accepts file paths)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, frame)

            detections = []

            if self.roboflow_client:
                # Use inference-sdk
                for model_id in self.roboflow_model_ids:
                    try:
                        result = self.roboflow_client.infer(tmp_path, model_id=model_id)
                        predictions = result.get("predictions", [])

                        for pred in predictions:
                            x = float(pred.get("x", 0))
                            y = float(pred.get("y", 0))
                            w = float(pred.get("width", 0))
                            h = float(pred.get("height", 0))

                            x1 = x - w / 2
                            y1 = y - h / 2
                            x2 = x + w / 2
                            y2 = y + h / 2

                            raw_class = pred.get("class", "").lower()
                            class_name = self.class_mapping.get(raw_class, raw_class)
                            confidence = float(pred.get("confidence", 0.0))

                            if confidence >= self.confidence_threshold:
                                detections.append(Detection(
                                    bbox=[x1, y1, x2, y2],
                                    class_name=class_name,
                                    confidence=confidence
                                ))
                    except Exception as e:
                        logger.error(f"Roboflow SDK inference failed for model {model_id}: {e}")
                        continue
            else:
                # Fallback to raw HTTP if SDK not available
                import base64
                import httpx

                with open(tmp_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")

                with httpx.Client(timeout=15.0) as client:
                    for model_id in self.roboflow_model_ids:
                        url = f"https://detect.roboflow.com/{model_id}"
                        params = {"api_key": self.roboflow_api_key}
                        response = client.post(
                            url, params=params, content=img_b64,
                            headers={"Content-Type": "application/x-www-form-urlencoded"}
                        )
                        if response.status_code != 200:
                            logger.error(f"Roboflow API ({model_id}) error {response.status_code}")
                            continue

                        data = response.json()
                        for pred in data.get("predictions", []):
                            x = float(pred.get("x", 0))
                            y = float(pred.get("y", 0))
                            w = float(pred.get("width", 0))
                            h = float(pred.get("height", 0))
                            x1, y1 = x - w / 2, y - h / 2
                            x2, y2 = x + w / 2, y + h / 2
                            raw_class = pred.get("class", "").lower()
                            class_name = self.class_mapping.get(raw_class, raw_class)
                            confidence = float(pred.get("confidence", 0.0))
                            if confidence >= self.confidence_threshold:
                                detections.append(Detection(
                                    bbox=[x1, y1, x2, y2],
                                    class_name=class_name,
                                    confidence=confidence
                                ))

            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            return detections
        except Exception as e:
            logger.error(f"Roboflow API inference failed: {e}")
            return []

    def detect(self, frame) -> List[Detection]:
        """Run detection on a single frame."""
        if self.roboflow_enabled:
            return self._detect_roboflow(frame)

        if not self.model:
            return []
        try:
            results = self.model(
                frame,
                device=self.device,
                conf=self.confidence_threshold,
                iou=getattr(settings, "NMS_THRESHOLD", 0.45),
                verbose=False
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    det = Detection(
                        bbox=box.xyxy[0].tolist(),
                        class_name=r.names[int(box.cls[0])],
                        confidence=float(box.conf[0]),
                    )
                    detections.append(det)
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def detect_with_sahi(self, frame, slice_size: int = 640, overlap: float = 0.2) -> List[Detection]:
        """Run SAHI sliced inference for small object detection."""
        if self.roboflow_enabled:
            logger.warning("SAHI is not recommended when Roboflow API is enabled due to API rate limits and latency. Falling back to standard Roboflow detection.")
            return self.detect(frame)
            
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_prediction
            import numpy as np

            if not self.model_path:
                return []

            if self.sahi_model is None:
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=self.model_path,
                    device=self.device,
                    confidence_threshold=self.confidence_threshold,
                )

            from sahi.predict import get_sliced_prediction
            result = get_sliced_prediction(
                frame,
                self.sahi_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
            )

            detections = []
            for pred in result.object_prediction_list:
                bbox = pred.bbox.to_xyxy()
                det = Detection(
                    bbox=bbox,
                    class_name=pred.category.name,
                    confidence=pred.score.value,
                )
                detections.append(det)
            return detections
        except ImportError:
            logger.warning("sahi not installed, falling back to standard detection")
            return self.detect(frame)

    def process_video(self, video_path: str, callback: Optional[Callable] = None, frame_skip: int = 1):
        """Process a video file frame by frame."""
        try:
            import cv2
        except ImportError:
            logger.error("opencv not installed")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                start = time.time()
                detections = self.detect(frame)
                elapsed = time.time() - start
                self.current_fps = 1.0 / max(elapsed, 0.001)
                self.frame_count += 1

                if callback:
                    callback(frame_idx, frame, detections, self.current_fps)

            frame_idx += 1

        cap.release()

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "model": self.model_path,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "fps": round(self.current_fps, 1),
            "frames_processed": self.frame_count,
        }
