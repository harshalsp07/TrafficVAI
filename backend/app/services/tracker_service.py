"""Object tracking service with ByteTrack-inspired cascade matching."""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Single tracked object with Kalman-like velocity prediction."""
    track_id: int
    bbox: List[float]  # [xmin, ymin, xmax, ymax]
    class_name: str
    confidence: float
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    speed: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)  # Estimated (dx, dy) per frame
    state: str = "tentative"  # tentative, confirmed, lost

    def __post_init__(self):
        self.class_votes = {self.class_name: 1}

    @property
    def centroid(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def is_stationary(self, threshold_pixels: float = 5.0) -> bool:
        """Check if track has been stationary over the last 5 trajectory points."""
        if len(self.trajectory) < 5:
            return False
        recent = self.trajectory[-5:]
        variance_x = sum((p[0] - recent[0][0])**2 for p in recent) / len(recent)
        variance_y = sum((p[1] - recent[0][1])**2 for p in recent) / len(recent)
        return (variance_x + variance_y) < threshold_pixels**2

    def predict(self) -> None:
        """Predict next bbox position using estimated velocity."""
        dx, dy = self.velocity
        self.bbox[0] += dx
        self.bbox[2] += dx
        self.bbox[1] += dy
        self.bbox[3] += dy
        # Age increases, update time increases
        self.time_since_update += 1
        self.age += 1

    def update_state(self, bbox: List[float], class_name: str, confidence: float) -> None:
        """Update track state with a new detection."""
        # Calculate velocity before updating centroid
        old_centroid = self.centroid
        new_centroid = (
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
        )
        
        # Exponential moving average for velocity to smooth noise
        alpha = 0.7
        dx = new_centroid[0] - old_centroid[0]
        dy = new_centroid[1] - old_centroid[1]
        self.velocity = (
            alpha * dx + (1 - alpha) * self.velocity[0],
            alpha * dy + (1 - alpha) * self.velocity[1],
        )

        self.bbox = bbox
        
        # Class majority voting logic
        if not hasattr(self, "class_votes"):
            self.class_votes = {}
        self.class_votes[class_name] = self.class_votes.get(class_name, 0) + 1
        self.class_name = max(self.class_votes, key=self.class_votes.get)

        self.confidence = confidence
        self.trajectory.append(new_centroid)
        if len(self.trajectory) > 100:
            self.trajectory = self.trajectory[-100:]
            
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        
        if self.state == "tentative" and self.hits >= 2:
            self.state = "confirmed"

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "trajectory": self.trajectory,
            "age": self.age,
            "speed": self.speed,
            "centroid": list(self.centroid),
            "state": self.state,
        }


def _iou(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


class TrackerService:
    """ByteTrack-inspired multi-object tracker."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 90, min_hits: int = 2,
                 high_conf_thresh: float = 0.5, low_conf_thresh: float = 0.1):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh

    def update(self, detections: List[Dict]) -> List[Track]:
        """Update tracker using the ByteTrack two-stage matching logic."""
        # 1. Split detections into high and low confidence sets
        high_conf_dets = []
        low_conf_dets = []
        
        for det in detections:
            conf = det.get("confidence", 0.0)
            if conf >= self.high_conf_thresh:
                high_conf_dets.append(det)
            elif conf >= self.low_conf_thresh:
                low_conf_dets.append(det)

        # 2. Kalman-like predict step for all existing tracks
        for track in self.tracks:
            track.predict()

        # 3. First Association: High-confidence detections with existing tracks
        unmatched_tracks, matched_dets_high = self._associate(self.tracks, high_conf_dets, self.iou_threshold)
        
        # 4. Second Association: Low-confidence detections with unmatched tracks
        # Note: we only match with confirmed/tentative tracks that were not updated in this frame
        tracks_for_second_association = [t for t in unmatched_tracks if t.state in ("confirmed", "tentative")]
        unmatched_tracks_second, matched_dets_low = self._associate(tracks_for_second_association, low_conf_dets, self.iou_threshold + 0.1) # higher threshold for low confidence
        
        # Keep track of which tracks got matched in first vs second stage
        updated_tracks = []
        
        # Apply updates for matched high-confidence detections
        for track_idx, det_idx in matched_dets_high.items():
            track = self.tracks[track_idx]
            det = high_conf_dets[det_idx]
            track.update_state(det["bbox"], det["class_name"], det["confidence"])
            updated_tracks.append(track)

        # Apply updates for matched low-confidence detections
        for track_idx, det_idx in matched_dets_low.items():
            # Find the track object in the main list
            track_to_update = tracks_for_second_association[track_idx]
            det = low_conf_dets[det_idx]
            track_to_update.update_state(det["bbox"], det["class_name"], det["confidence"])
            updated_tracks.append(track_to_update)

        # 5. Handle unmatched high-confidence detections: Init new tracks
        matched_high_indices = set(matched_dets_high.values())
        for idx, det in enumerate(high_conf_dets):
            if idx not in matched_high_indices:
                new_track = self._create_track(det)
                updated_tracks.append(new_track)

        # 6. Update track list: Keep tracks that are within max_age limit
        # Lost tracks are kept in the list for a while so they can be re-associated
        all_tracks_to_keep = []
        for track in self.tracks:
            if track in updated_tracks:
                all_tracks_to_keep.append(track)
            elif track.time_since_update <= self.max_age:
                track.state = "lost"
                all_tracks_to_keep.append(track)

        self.tracks = all_tracks_to_keep
        return self.tracks

    def _associate(self, tracks: List[Track], detections: List[Dict], threshold: float) -> Tuple[List[Track], Dict[int, int]]:
        """Perform association matching via IoU."""
        if not tracks or not detections:
            return list(tracks), {}

        # Build IoU matrix
        iou_matrix = []
        for track in tracks:
            row = [_iou(track.bbox, det["bbox"]) for det in detections]
            iou_matrix.append(row)

        matched_tracks = {}
        matched_dets = set()
        
        # Greedy matching
        for _ in range(min(len(tracks), len(detections))):
            best_iou = 0.0
            best_t = -1
            best_d = -1
            for t_idx in range(len(tracks)):
                if t_idx in matched_tracks:
                    continue
                for d_idx in range(len(detections)):
                    if d_idx in matched_dets:
                        continue
                    if iou_matrix[t_idx][d_idx] > best_iou:
                        best_iou = iou_matrix[t_idx][d_idx]
                        best_t = t_idx
                        best_d = d_idx
            
            if best_iou >= threshold:
                matched_tracks[best_t] = best_d
                matched_dets.add(best_d)
            else:
                break

        unmatched_tracks = [tracks[i] for i in range(len(tracks)) if i not in matched_tracks]
        return unmatched_tracks, matched_tracks

    def _create_track(self, detection: Dict) -> Track:
        track = Track(
            track_id=self.next_id,
            bbox=detection.get("bbox", [0.0, 0.0, 0.0, 0.0]),
            class_name=detection.get("class_name", "unknown"),
            confidence=detection.get("confidence", 0.0),
        )
        track.trajectory.append(track.centroid)
        self.tracks.append(track)
        self.next_id += 1
        return track

    def get_track(self, track_id: int) -> Optional[Track]:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
