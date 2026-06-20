import logging
import numpy as np
import time
import cv2
import re
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel

from app.config import settings
from app.services.preprocessing import FramePreprocessor, FrameQuality
from app.services.inference_engine import InferenceEngine, Detection
from app.services.tracker_service import TrackerService, Track
from app.services.violation_detector import ViolationDetector, Violation
from app.services.anpr_service import ANPRService
from app.services.evidence_generator import EvidenceGenerator, EvidencePackage
from app.services.speed_estimator import SpeedEstimator
from app.services.dedup_engine import DedupEngine

logger = logging.getLogger(__name__)

class MicroClassifier:
    """Lightweight classifier wrapper for windshield/traffic light crops."""

    def __init__(self, name: str, model_path: str, device: str = "cpu"):
        self.name = name
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load()

    def _load(self):
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms
            from PIL import Image

            # Define transform
            self.transform = transforms.Compose([
                transforms.Resize((224, 224) if self.name == "seatbelt" else (64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            if self.name == "seatbelt":
                # MobileNetV3 Small binary classifier
                self.model = models.mobilenet_v3_small(pretrained=False)
                in_features = getattr(self.model.classifier[3], "in_features", 1024)
                self.model.classifier[3] = nn.Linear(in_features, 2)
            else:
                # ResNet18 3-class traffic light classifier (red, yellow, green)
                self.model = models.resnet18(pretrained=False)
                self.model.fc = nn.Linear(self.model.fc.in_features, 3)

            if torch.cuda.is_available() and "cuda" in self.device:
                self.model = self.model.cuda()
                
            # Try to load state dict if weights file exists
            import os
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                logger.info(f"Loaded micro-classifier {self.name} weights from {self.model_path}")
            else:
                logger.warning(f"Micro-classifier {self.name} weights not found at {self.model_path}. Using random mock predictions.")
                self.model = None
        except Exception as e:
            logger.warning(f"Could not load micro-classifier {self.name}: {e}. Falling back to mock prediction.")
            self.model = None

    def __call__(self, crop: np.ndarray) -> Tuple[int, float]:
        """Classify a cropped image region. Returns (class_id, probability)."""
        if self.model is None or crop is None or crop.size == 0:
            # Safe mock behavior: return seatbelt_on (0) or green traffic light (2)
            # to prevent false violations in production/demo when weights are missing.
            if self.name == "seatbelt":
                # 0: seatbelt_on, 1: seatbelt_off
                return 0, 1.0
            else:
                # 0: red, 1: yellow, 2: green
                return 2, 1.0

        try:
            import torch
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = self.transform(img).unsqueeze(0)
            if torch.cuda.is_available() and "cuda" in self.device:
                tensor = tensor.cuda()

            with torch.no_grad():
                output = self.model(tensor)
                prob = torch.softmax(output, dim=1)
                val, idx = torch.max(prob, dim=1)
                return int(idx.item()), float(val.item())
        except Exception as e:
            logger.error(f"Error during classifier inference: {e}")
            return 0, 0.5


class PipelineResult:
    def __init__(self, tracks: List[Track], violations: List[Violation], 
                 evidence_packages: List[EvidencePackage], quality: FrameQuality):
        self.tracks = tracks
        self.violations = violations
        self.evidence_packages = evidence_packages
        self.quality = quality


class UnifiedPipeline:
    """Production-grade unified multi-task cascade pipeline."""

    def __init__(self, ocr_engine: str = "easyocr"):
        self.detector = InferenceEngine()
        
        # Load the unified YOLOv11m detector model
        # Using model fallback chain if the path isn't fully ready
        self.detector.load_model(settings.MODEL_PATH, settings.DEVICE)
        
        # Tracker Service (ByteTrack-inspired)
        self.tracker = TrackerService(
            iou_threshold=settings.NMS_THRESHOLD,
            high_conf_thresh=settings.CONFIDENCE_THRESHOLD,
            low_conf_thresh=0.1
        )
        
        self.preprocessor = FramePreprocessor()
        self.violation_detector = ViolationDetector(config={
            "helmet_confidence": settings.HELMET_CONFIDENCE_THRESHOLD,
            "speed_limit": settings.SPEED_LIMIT_KMH,
            "wrong_side_cosine": settings.WRONG_SIDE_COSINE_THRESHOLD,
            "parking_duration": settings.PARKING_DURATION_SECONDS,
            "min_consecutive_frames": settings.MIN_CONSECUTIVE_FRAMES
        })
        
        self.anpr = ANPRService(ocr_engine=settings.OCR_ENGINE)
        self.evidence_gen = EvidenceGenerator(evidence_dir=settings.EVIDENCE_DIR)
        
        # Speed Estimator
        self.speed_estimator = SpeedEstimator()
        
        # Micro-classifiers
        self.seatbelt_cls = MicroClassifier("seatbelt", settings.SEATBELT_MODEL_PATH, settings.DEVICE)
        self.traffic_light_cls = MicroClassifier("traffic_light", settings.TRAFFIC_LIGHT_MODEL_PATH, settings.DEVICE)
        
        self.frame_counter = 0
        self.frame_skip = settings.FRAME_SKIP
        
        # plate-based deduplication history: (license_plate, violation_type) -> timestamp
        self._plate_violation_history: Dict[Tuple[str, str], float] = {}
        self.dedup_engine = DedupEngine()

    def process_frame(self, frame: np.ndarray, camera_config: Dict) -> PipelineResult:
        """Execute the full 7-stage cascade on a single frame."""
        self.frame_counter += 1

        # Stage 1: Quality Assessment and Preprocessing
        quality = self.preprocessor.assess_quality(frame)
        enhanced = self.preprocessor.enhance(frame, quality)

        # Stage 2: Unified YOLO Detection (or skip + tracker-only prediction)
        detections = []
        is_inference_frame = (self.frame_counter % self.frame_skip == 0)
        
        # If tracker has low confidence, force inference to recover lost tracks
        tracker_confidence = sum(t.confidence for t in self.tracker.tracks) / max(1, len(self.tracker.tracks))
        force_inference = (tracker_confidence < 0.4 and len(self.tracker.tracks) > 0)
        
        if is_inference_frame or force_inference:
            if settings.SAHI_ENABLED or camera_config.get("sahi_enabled", False):
                detections = self.detector.detect_with_sahi(
                    enhanced,
                    slice_size=settings.SAHI_SLICE_HEIGHT,
                    overlap=settings.SAHI_OVERLAP_RATIO
                )
            else:
                detections = self.detector.detect(enhanced)
        
        # Convert Detection objects to lists/dicts for tracker/violation logic
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                "bbox": det.bbox,
                "class_name": det.class_name,
                "confidence": det.confidence
            })

        # Stage 3: ByteTrack Multi-Object Tracking
        # If inference was skipped, detections is empty and tracker will predict positions
        tracks = self.tracker.update(formatted_detections)

        # Determine effective processing FPS for speed estimation
        processing_fps = camera_config.get("processing_fps")
        if processing_fps is not None:
            effective_fps = float(processing_fps)
        else:
            camera_fps = camera_config.get("fps", 30.0)
            effective_fps = camera_fps / max(1, self.frame_skip)

        # Update track speeds
        for track in tracks:
            if len(track.trajectory) >= 2:
                track.speed = self.speed_estimator.estimate_speed(
                    track.trajectory,
                    fps=effective_fps,
                    window=5
                )

        # Stage 4: Run Micro-Classifiers for Traffic Light state
        # Find traffic_light detections to update signal state in config
        signal_state = camera_config.get("signal_state", "green")
        traffic_lights = [d for d in formatted_detections if d["class_name"] == "traffic_light"]
        
        if traffic_lights:
            # Take the largest or highest confidence traffic light
            best_light = max(traffic_lights, key=lambda x: x["confidence"])
            # Crop traffic light box
            lx1, ly1, lx2, ly2 = map(int, best_light["bbox"])
            h_img, w_img, _ = enhanced.shape
            lx1, lx2 = max(0, lx1), min(w_img, lx2)
            ly1, ly2 = max(0, ly1), min(h_img, ly2)
            
            if lx2 - lx1 > 5 and ly2 - ly1 > 5:
                light_crop = enhanced[ly1:ly2, lx1:lx2]
                light_idx, light_prob = self.traffic_light_cls(light_crop)
                # 0: red, 1: yellow, 2: green
                if light_prob > 0.8:
                    states = ["red", "yellow", "green"]
                    signal_state = states[light_idx]
                    camera_config["signal_state"] = signal_state

        # Stage 5: Violation Detection Rule Engine
        classifiers = {
            "seatbelt": self.seatbelt_cls
        }
        violations = self.violation_detector.check_all_violations(
            enhanced, formatted_detections, tracks, camera_config, classifiers
        )

        # Stage 6: ANPR Pipeline (runs OCR ONLY for vehicles with violations)
        evidence_packages = []
        filtered_violations = []
        
        now = time.time()
        # Clean up plate history older than 300 seconds to prevent leaks
        self._plate_violation_history = {
            k: t for k, t in self._plate_violation_history.items() if now - t < 300.0
        }

        import re  # Safe to import here, but we will also ensure it is at top (actually, importing at top is better, let's do that)
        camera_id = camera_config.get("id", "default_camera")

        for v in violations:
            vehicle_track = self.tracker.get_track(v.track_id)
            if vehicle_track:
                # Find overlapping license plate
                plate_bbox = self.anpr.detect_plate(vehicle_track.bbox, formatted_detections)
                if plate_bbox:
                    px1, py1, px2, py2 = map(int, plate_bbox)
                    h_img, w_img, _ = enhanced.shape
                    px1, px2 = max(0, px1), min(w_img, px2)
                    py1, py2 = max(0, py1), min(h_img, py2)
                    
                    if px2 - px1 > 5 and py2 - py1 > 5:
                        plate_crop = enhanced[py1:py2, px1:px2]
                        plate_text, plate_conf = self.anpr.read_plate(plate_crop)
                        
                        v.details["license_plate"] = plate_text
                        v.details["plate_confidence"] = plate_conf
                        v.details["plate_bbox"] = plate_bbox
                
                # Crop vehicle region for perceptual hashing
                vx1, vy1, vx2, vy2 = map(int, vehicle_track.bbox)
                h_img, w_img, _ = enhanced.shape
                vx1, vx2 = max(0, vx1), min(w_img, vx2)
                vy1, vy2 = max(0, vy1), min(h_img, vy2)
                vehicle_crop = enhanced[vy1:vy2, vx1:vx2] if (vx2 - vx1 > 0 and vy2 - vy1 > 0) else None

                license_plate = v.details.get("license_plate", "UNKNOWN")

                # Plate-based de-duplication (Layer 3)
                clean_plate = re.sub(r'[^A-Z0-9]', '', license_plate.upper()) if license_plate else ""
                is_duplicate_plate = False
                
                if clean_plate and clean_plate != "UNKNOWN" and len(clean_plate) >= 4:
                    key = (camera_id, clean_plate, v.violation_type)
                    if key in self._plate_violation_history:
                        logger.info(f"[DEDUPLICATION] Suppressed duplicate violation of type '{v.violation_type}' for plate '{clean_plate}' on camera '{camera_id}'")
                        is_duplicate_plate = True
                    else:
                        self._plate_violation_history[key] = now

                if is_duplicate_plate:
                    continue

                # Perceptual hash deduplication (Layer 4)
                should_rec, hash_str = self.dedup_engine.should_record(camera_id, v.violation_type, vehicle_crop)
                if not should_rec:
                    continue
                v.details["vehicle_hash"] = hash_str

                # Format violation details for evidence generator
                violation_dict = {
                    "type": v.violation_type,
                    "confidence": v.confidence,
                    "bbox": v.bbox,
                    "track_id": v.track_id,
                    "vehicle_class": v.vehicle_class,
                    "license_plate": license_plate,
                    "plate_confidence": v.details.get("plate_confidence", 0.0),
                    "plate_bbox": v.details.get("plate_bbox"),
                    "vehicle_hash": hash_str
                }

                # Stage 7: Evidence Generation
                ev_package = self.evidence_gen.generate(
                    frame, enhanced, violation_dict, tracks, camera_config
                )
                evidence_packages.append(ev_package)
                filtered_violations.append(v)

        return PipelineResult(tracks, filtered_violations, evidence_packages, quality)
