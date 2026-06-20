"""Violation detection service."""
import math
import logging
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """Detected violation."""
    violation_type: str
    vehicle_class: str
    track_id: int
    confidence: float
    bbox: List[float]
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict:
        return {
            "violation_type": self.violation_type,
            "vehicle_class": self.vehicle_class,
            "track_id": self.track_id,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "details": self.details,
        }


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / max(union, 1e-6)


def _cosine_similarity(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Compute cosine similarity between two 2D vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    return dot / max(mag1 * mag2, 1e-6)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Extract attribute from object, dictionary, or Detection dataclass."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class ViolationDetector:
    """Multi-violation detection engine with spatial-temporal checks."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.helmet_threshold = self.config.get("helmet_confidence", 0.6)
        self.speed_limit = self.config.get("speed_limit", 60)
        self.wrong_side_threshold = self.config.get("wrong_side_cosine", -0.5)
        self.parking_duration = self.config.get("parking_duration", 300)
        self.min_frames = self.config.get("min_consecutive_frames", 3)
        
        # Temporal violation counters
        self._helmet_counter: Dict[int, int] = {}
        self._triple_riding_counter: Dict[int, int] = {}
        self._seatbelt_counter: Dict[int, int] = {}
        self._stop_line_counter: Dict[int, int] = {}
        self._parking_tracker: Dict[int, Dict] = {}
        
        # History of recently logged violations to avoid duplication (track_id, violation_type) -> timestamp
        self._violation_history: Dict[Tuple[int, str], float] = {}

    def check_helmet(self, detections: List[Any], tracks: List[Any]) -> List[Violation]:
        """Check for helmet non-compliance on motorcycles."""
        violations = []
        
        # Extract detections
        motorcycles = [d for d in detections if _get_attr(d, "class_name") == "motorcycle"]
        riders = [d for d in detections if _get_attr(d, "class_name") == "rider"]
        helmets = [d for d in detections if _get_attr(d, "class_name") == "helmet"]

        for mc in motorcycles:
            mc_bbox = _get_attr(mc, "bbox")
            if not mc_bbox:
                continue
                
            associated_riders = [r for r in riders if _iou(_get_attr(r, "bbox"), mc_bbox) > 0.15]

            for rider in associated_riders:
                rider_bbox = _get_attr(rider, "bbox")
                if not rider_bbox:
                    continue
                    
                # The head region is typically in the top 30% of the rider bounding box
                head_roi = [
                    rider_bbox[0],
                    rider_bbox[1],
                    rider_bbox[2],
                    rider_bbox[1] + (rider_bbox[3] - rider_bbox[1]) * 0.3,
                ]
                
                # Check if there is an overlapping helmet
                has_helmet = any(
                    _iou(_get_attr(h, "bbox"), head_roi) > 0.2 and _get_attr(h, "confidence", 0) > self.helmet_threshold
                    for h in helmets
                )
                
                track_id = _get_attr(rider, "track_id", 0)
                if not track_id:
                    # Check if the motorcycle itself has a track_id we can attribute it to
                    track_id = _get_attr(mc, "track_id", 0)

                if not has_helmet:
                    self._helmet_counter[track_id] = self._helmet_counter.get(track_id, 0) + 1
                    if self._helmet_counter[track_id] >= self.min_frames:
                        violations.append(Violation(
                            violation_type="helmet",
                            vehicle_class="motorcycle",
                            track_id=track_id,
                            confidence=float(_get_attr(rider, "confidence", 0.8)),
                            bbox=mc_bbox,
                            details={"rider_count": len(associated_riders), "helmet_found": False},
                        ))
                else:
                    self._helmet_counter.pop(track_id, None)

        return violations

    def check_triple_riding(self, detections: List[Any], tracks: List[Any]) -> List[Violation]:
        """Check for triple riding (3+ riders/pedestrians associated with a motorcycle)."""
        violations = []
        motorcycles = [d for d in detections if _get_attr(d, "class_name") == "motorcycle"]
        riders = [d for d in detections if _get_attr(d, "class_name") in ("rider", "pedestrian")]

        for mc in motorcycles:
            mc_bbox = _get_attr(mc, "bbox")
            if not mc_bbox:
                continue
                
            # Count how many rider/pedestrian bounding boxes overlap with the motorcycle
            associated_riders = [r for r in riders if _iou(_get_attr(r, "bbox"), mc_bbox) > 0.15]
            rider_count = len(associated_riders)
            
            track_id = _get_attr(mc, "track_id", 0)
            if not track_id:
                continue

            if rider_count >= 3:
                self._triple_riding_counter[track_id] = self._triple_riding_counter.get(track_id, 0) + 1
                if self._triple_riding_counter[track_id] >= 3: # Temporal threshold
                    violations.append(Violation(
                        violation_type="triple_riding",
                        vehicle_class="motorcycle",
                        track_id=track_id,
                        confidence=float(min(0.98, 0.7 + rider_count * 0.1)),
                        bbox=mc_bbox,
                        details={"rider_count": rider_count},
                    ))
            else:
                self._triple_riding_counter.pop(track_id, None)

        return violations

    def check_seatbelt(self, frame: np.ndarray, detections: List[Any], tracks: List[Any], 
                       seatbelt_classifier: Optional[Any] = None) -> List[Violation]:
        """Check for seatbelt non-compliance in cars, trucks, or buses."""
        violations = []
        vehicles = [t for t in tracks if _get_attr(t, "class_name") in ("car", "truck", "bus")]

        for v in vehicles:
            bbox = _get_attr(v, "bbox")
            track_id = _get_attr(v, "track_id", 0)
            if not bbox or not track_id:
                continue

            # Suppression check: if this vehicle overlaps with a motorcycle, rider, pedestrian, or bicycle,
            # suppress seatbelt detection to avoid false charges (e.g. rider in windshield crop or class swap).
            is_suppressed = False
            for det in detections:
                det_class = _get_attr(det, "class_name")
                if det_class in ("motorcycle", "rider", "pedestrian", "bicycle", "auto_rickshaw"):
                    det_bbox = _get_attr(det, "bbox")
                    if det_bbox and _iou(bbox, det_bbox) > 0.15:
                        is_suppressed = True
                        logger.info(f"Seatbelt check suppressed for track {track_id} due to overlap with {det_class}")
                        break
            
            if is_suppressed:
                self._seatbelt_counter.pop(track_id, None)
                continue
                
            # Define windshield crop area (top 40% of vehicle box)
            x1, y1, x2, y2 = map(int, bbox)
            w = x2 - x1
            h = y2 - y1
            
            # Windshield bounding box
            wx1, wy1, wx2, wy2 = x1, y1, x2, int(y1 + h * 0.45)
            
            # Crop windshield
            h_img, w_img, _ = frame.shape
            wx1, wx2 = max(0, wx1), min(w_img, wx2)
            wy1, wy2 = max(0, wy1), min(h_img, wy2)
            
            if wx2 - wx1 <= 10 or wy2 - wy1 <= 10:
                continue
                
            # Windshield crop
            windshield_crop = frame[wy1:wy2, wx1:wx2]
            
            # Perform seatbelt classification
            # If classifier is not loaded, we mock it with a safe default or lightweight check
            is_seatbelt_off = False
            cls_confidence = 0.0
            
            if seatbelt_classifier is not None:
                try:
                    # Let's assume class 1 is "seatbelt_off" and class 0 is "seatbelt_on"
                    label_idx, prob = seatbelt_classifier(windshield_crop)
                    is_seatbelt_off = (label_idx == 1)
                    cls_confidence = prob
                except Exception as e:
                    logger.error(f"Seatbelt classifier error: {e}")
                    is_seatbelt_off = False
            else:
                # Mock fallback: simulate seatbelt off for training demo validation
                # In real scenario, this runs the classifier
                is_seatbelt_off = False

            if is_seatbelt_off and cls_confidence > 0.7:
                self._seatbelt_counter[track_id] = self._seatbelt_counter.get(track_id, 0) + 1
                if self._seatbelt_counter[track_id] >= self.min_frames:
                    violations.append(Violation(
                        violation_type="seatbelt",
                        vehicle_class=_get_attr(v, "class_name", "car"),
                        track_id=track_id,
                        confidence=float(cls_confidence * _get_attr(v, "confidence", 1.0)),
                        bbox=bbox,
                        details={"seatbelt_on": False, "windshield_box": [wx1, wy1, wx2, wy2]},
                    ))
            else:
                self._seatbelt_counter.pop(track_id, None)

        return violations

    def check_speed(self, tracks: List[Any], speed_limit: Optional[float] = None) -> List[Violation]:
        """Check for speed violations."""
        limit = speed_limit or self.speed_limit
        violations = []
        motorized = getattr(settings, "MOTORIZED_VEHICLES", ["car", "truck", "bus", "motorcycle", "auto_rickshaw", "van"])
        for track in tracks:
            class_name = _get_attr(track, "class_name", "vehicle")
            if class_name not in motorized:
                continue
            speed = _get_attr(track, "speed", 0.0)
            if speed > limit:
                violations.append(Violation(
                    violation_type="speed",
                    vehicle_class=class_name,
                    track_id=_get_attr(track, "track_id", 0),
                    confidence=float(min(0.98, 0.7 + (speed - limit) / limit * 0.3)),
                    bbox=_get_attr(track, "bbox", [0, 0, 0, 0]),
                    details={"speed_kmh": round(speed, 1), "limit_kmh": limit},
                ))
        return violations

    def check_wrong_side(self, tracks: List[Any], lane_direction: Tuple[float, float]) -> List[Violation]:
        """Check for wrong-side driving using trajectory direction vs lane direction."""
        violations = []
        motorized = getattr(settings, "MOTORIZED_VEHICLES", ["car", "truck", "bus", "motorcycle", "auto_rickshaw", "van"])
        for track in tracks:
            class_name = _get_attr(track, "class_name", "vehicle")
            if class_name not in motorized:
                continue
            trajectory = _get_attr(track, "trajectory", [])
            if len(trajectory) < 3:
                continue
                
            # Compute movement vector between last points
            dx = trajectory[-1][0] - trajectory[-3][0]
            dy = trajectory[-1][1] - trajectory[-3][1]
            if abs(dx) < 2 and abs(dy) < 2:
                continue
                
            movement = (dx, dy)
            cos_sim = _cosine_similarity(movement, lane_direction)
            
            # If moving opposite to lane direction
            if cos_sim < self.wrong_side_threshold:
                violations.append(Violation(
                    violation_type="wrong_side",
                    vehicle_class=class_name,
                    track_id=_get_attr(track, "track_id", 0),
                    confidence=float(min(0.95, abs(cos_sim))),
                    bbox=_get_attr(track, "bbox", [0, 0, 0, 0]),
                    details={"cosine_similarity": round(cos_sim, 3), "lane_direction": lane_direction},
                ))
        return violations

    def check_red_light(self, tracks: List[Any], stop_line_y: float, signal_state: str) -> List[Violation]:
        """Check for red-light violation (crossing stop line and proceeding through junction)."""
        violations = []
        if signal_state.lower() != "red":
            return violations

        motorized = getattr(settings, "MOTORIZED_VEHICLES", ["car", "truck", "bus", "motorcycle", "auto_rickshaw", "van"])
        for track in tracks:
            class_name = _get_attr(track, "class_name", "vehicle")
            if class_name not in motorized:
                continue
            trajectory = _get_attr(track, "trajectory", [])
            if len(trajectory) < 2:
                continue
                
            prev_centroid = trajectory[-2]
            curr_centroid = trajectory[-1]
            
            # Centroid crosses the stop line and continues moving down (or across)
            if prev_centroid[1] < stop_line_y and curr_centroid[1] >= stop_line_y:
                violations.append(Violation(
                    violation_type="red_light",
                    vehicle_class=class_name,
                    track_id=_get_attr(track, "track_id", 0),
                    confidence=float(0.92 * _get_attr(track, "confidence", 1.0)),
                    bbox=_get_attr(track, "bbox", [0, 0, 0, 0]),
                    details={"signal_state": signal_state, "stop_line_y": stop_line_y},
                ))
        return violations

    def check_stop_line_violation(self, tracks: List[Any], stop_line_y: float, signal_state: str) -> List[Violation]:
        """Check if a vehicle stopped past the stop line without going through the junction."""
        violations = []
        if signal_state.lower() != "red":
            return violations

        motorized = getattr(settings, "MOTORIZED_VEHICLES", ["car", "truck", "bus", "motorcycle", "auto_rickshaw", "van"])
        for track in tracks:
            class_name = _get_attr(track, "class_name", "vehicle")
            if class_name not in motorized:
                continue
            bbox = _get_attr(track, "bbox", [0, 0, 0, 0])
            bottom_y = bbox[3]
            trajectory = _get_attr(track, "trajectory", [])
            if len(trajectory) < 3:
                continue
                
            # If bottom of vehicle is past stop line, but vehicle is nearly stationary
            dx = trajectory[-1][0] - trajectory[-3][0]
            dy = trajectory[-1][1] - trajectory[-3][1]
            speed = math.sqrt(dx**2 + dy**2)
            
            track_id = _get_attr(track, "track_id", 0)
            if not track_id:
                continue

            # Vehicle bottom is past stop line, but centroid is still before junction, and speed is extremely slow
            if bottom_y > stop_line_y and trajectory[-1][1] < stop_line_y + 40 and speed < 2.0:
                self._stop_line_counter[track_id] = self._stop_line_counter.get(track_id, 0) + 1
                if self._stop_line_counter[track_id] >= self.min_frames:
                    violations.append(Violation(
                        violation_type="stop_line",
                        vehicle_class=class_name,
                        track_id=track_id,
                        confidence=float(0.90 * _get_attr(track, "confidence", 1.0)),
                        bbox=bbox,
                        details={"signal_state": signal_state, "stop_line_y": stop_line_y, "speed": round(speed, 2)},
                    ))
            else:
                self._stop_line_counter.pop(track_id, None)
        return violations

    def check_illegal_parking(self, tracks: List[Any], zone_polygons: List[List[List[float]]], 
                              duration_threshold: Optional[float] = None) -> List[Violation]:
        """Check for illegal parking in no-parking zones (stationary in polygon for a duration)."""
        threshold = duration_threshold or self.parking_duration
        violations = []
        motorized = getattr(settings, "MOTORIZED_VEHICLES", ["car", "truck", "bus", "motorcycle", "auto_rickshaw", "van"])

        for track in tracks:
            class_name = _get_attr(track, "class_name", "vehicle")
            if class_name not in motorized:
                continue
            track_id = _get_attr(track, "track_id", 0)
            centroid = _get_attr(track, "centroid", [0, 0])
            trajectory = _get_attr(track, "trajectory", [])

            if len(trajectory) < 5 or not track_id:
                continue

            # Check if centroid is inside any of the restricted polygons
            in_zone = False
            for poly in zone_polygons:
                # cv2.pointPolygonTest requires float32 array
                poly_np = np.array(poly, dtype=np.float32)
                dist = cv2.pointPolygonTest(poly_np, (float(centroid[0]), float(centroid[1])), False)
                if dist >= 0:
                    in_zone = True
                    break
                    
            if not in_zone:
                self._parking_tracker.pop(track_id, None)
                continue

            # Check if stationary (low variance in coordinates over last 5 points)
            recent = trajectory[-5:]
            variance_x = sum((p[0] - recent[0][0])**2 for p in recent) / len(recent)
            variance_y = sum((p[1] - recent[0][1])**2 for p in recent) / len(recent)
            is_stationary = (variance_x + variance_y) < 25.0  # within 5px box

            if is_stationary:
                if track_id not in self._parking_tracker:
                    self._parking_tracker[track_id] = {"start": time.time()}
                elapsed = time.time() - self._parking_tracker[track_id]["start"]
                if elapsed >= threshold:
                    violations.append(Violation(
                        violation_type="illegal_parking",
                        vehicle_class=class_name,
                        track_id=track_id,
                        confidence=float(min(0.95, 0.7 + elapsed / (threshold * 2) * 0.25)),
                        bbox=_get_attr(track, "bbox", [0, 0, 0, 0]),
                        details={"duration_seconds": round(elapsed, 1)},
                    ))
            else:
                self._parking_tracker.pop(track_id, None)

        return violations

    def check_all_violations(self, frame: np.ndarray, detections: List[Any], tracks: List[Any], 
                             camera_config: Dict, classifiers: Optional[Dict[str, Any]] = None) -> List[Violation]:
        """Orchestrate all 7 violation detection checks in parallel."""
        violations = []
        classifiers = classifiers or {}
        
        # Extract zone parameters from camera config
        stop_line_y = camera_config.get("stop_line_y", 350.0)
        signal_state = camera_config.get("signal_state", "green")
        lane_direction = camera_config.get("lane_direction", (0.0, 1.0)) # Default moving down
        no_parking_zones = camera_config.get("no_parking_zones", [])
        speed_limit = camera_config.get("speed_limit", self.speed_limit)

        # Run checks
        # 1. Helmet non-compliance
        violations += self.check_helmet(detections, tracks)
        
        # 2. Triple riding
        violations += self.check_triple_riding(detections, tracks)
        
        # 3. Wrong side driving
        violations += self.check_wrong_side(tracks, lane_direction)
        
        # 4. Speeding
        violations += self.check_speed(tracks, speed_limit)
        
        # 5. Red light violation
        violations += self.check_red_light(tracks, stop_line_y, signal_state)
        
        # 6. Stop line violation
        violations += self.check_stop_line_violation(tracks, stop_line_y, signal_state)
        
        # 7. Illegal parking
        if no_parking_zones:
            violations += self.check_illegal_parking(tracks, no_parking_zones)
            
        # 8. Seatbelt non-compliance
        seatbelt_model = classifiers.get("seatbelt")
        violations += self.check_seatbelt(frame, detections, tracks, seatbelt_model)

        # Apply deduplication
        deduplicated = self.deduplicate(violations)
        
        # Clean up stale counters for tracks that are no longer active
        active_track_ids = [_get_attr(t, "track_id", 0) for t in tracks if _get_attr(t, "track_id", 0)]
        self._cleanup_stale_counters(active_track_ids)

        return deduplicated

    def _cleanup_stale_counters(self, active_track_ids: List[int]) -> None:
        """Remove counters for track IDs that are no longer active to prevent memory leaks."""
        active_set = set(active_track_ids)
        
        # Cleanup helmet counter
        for tid in list(self._helmet_counter.keys()):
            if tid not in active_set:
                self._helmet_counter.pop(tid, None)
                
        # Cleanup triple riding counter
        for tid in list(self._triple_riding_counter.keys()):
            if tid not in active_set:
                self._triple_riding_counter.pop(tid, None)
                
        # Cleanup seatbelt counter
        for tid in list(self._seatbelt_counter.keys()):
            if tid not in active_set:
                self._seatbelt_counter.pop(tid, None)
                
        # Cleanup stop line counter
        for tid in list(self._stop_line_counter.keys()):
            if tid not in active_set:
                self._stop_line_counter.pop(tid, None)
                
        # Cleanup parking tracker
        for tid in list(self._parking_tracker.keys()):
            if tid not in active_set:
                self._parking_tracker.pop(tid, None)

    def deduplicate(self, violations: List[Violation], window_seconds: float = 10.0) -> List[Violation]:
        """Deduplicate violations for the same vehicle/rider within a sliding time window."""
        result = []
        now = time.time()
        
        # Clean up history older than window_seconds
        self._violation_history = {k: t for k, t in self._violation_history.items() if now - t < window_seconds}

        for v in violations:
            key = (v.track_id, v.violation_type)
            # If we haven't logged this type of violation for this track_id recently
            if key not in self._violation_history:
                self._violation_history[key] = now
                result.append(v)
            else:
                # Update timestamp to slide the window if it keeps happening
                self._violation_history[key] = now
                
        return result
