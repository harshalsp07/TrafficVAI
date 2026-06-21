"""Image annotation service for drawing bounding boxes and violation markers."""
import cv2
import numpy as np
import os
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

VIOLATION_COLORS = {
    "helmet_non_compliance": (0, 0, 255),      # Red
    "triple_riding": (0, 128, 255),             # Orange
    "seatbelt_non_compliance": (0, 255, 255),   # Yellow
    "speed_violation": (255, 0, 0),             # Blue
    "wrong_side_driving": (255, 0, 255),        # Magenta
    "red_light_running": (0, 0, 200),           # Dark Red
    "stop_line_violation": (128, 0, 128),       # Purple
    "illegal_parking": (0, 128, 0),             # Green
}

CLASS_COLORS = {
    "car": (255, 180, 0),
    "truck": (200, 100, 0),
    "motorcycle": (0, 200, 255),
    "bus": (0, 150, 200),
    "bicycle": (0, 255, 100),
    "pedestrian": (200, 0, 200),
    "rider": (0, 200, 200),
    "helmet": (0, 255, 0),
    "no_helmet": (0, 0, 255),
    "license_plate": (255, 255, 0),
    "traffic_light": (255, 255, 255),
}


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw YOLO detection bounding boxes on image."""
    annotated = image.copy()

    for det in detections:
        bbox = det.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = map(int, bbox[:4])
        class_name = det.get("class_name", "unknown")
        confidence = det.get("confidence", 0.0)

        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        if show_labels:
            label = f"{class_name} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return annotated


def draw_violations(
    image: np.ndarray,
    violations: List[Dict[str, Any]],
    thickness: int = 3,
) -> np.ndarray:
    """Draw violation markers with thick colored bounding boxes and labels."""
    annotated = image.copy()

    for v in violations:
        bbox = v.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = map(int, bbox[:4])
        vtype = v.get("violation_type", v.get("type", "unknown"))
        confidence = v.get("confidence", 0.0)

        color = VIOLATION_COLORS.get(vtype, (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"VIOLATION: {vtype.replace('_', ' ').title()}"
        conf_label = f"Confidence: {confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 12, th + 10)

        cv2.rectangle(annotated, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(
            annotated, label, (x1 + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        (cw, ch), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        conf_y = label_y + th + 10
        cv2.rectangle(annotated, (x1, conf_y - ch - 4), (x1 + cw + 8, conf_y + 4), (0, 0, 0), -1)
        cv2.putText(
            annotated, conf_label, (x1 + 4, conf_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        plate_text = v.get("license_plate") or v.get("details", {}).get("license_plate")
        if plate_text:
            plate_label = f"Plate: {plate_text}"
            (pw, ph), _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            plate_y = conf_y + ch + 14
            cv2.rectangle(annotated, (x1, plate_y - ph - 4), (x1 + pw + 8, plate_y + 4), (0, 0, 0), -1)
            cv2.putText(
                annotated, plate_label, (x1 + 4, plate_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
            )

    return annotated


def annotate_image(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    violations: List[Dict[str, Any]],
) -> np.ndarray:
    """Full annotation: draw detections then overlay violations."""
    annotated = draw_bounding_boxes(image, detections)
    annotated = draw_violations(annotated, violations)
    return annotated


def save_annotated(
    image: np.ndarray,
    evidence_dir: str = "./evidence/annotated",
    prefix: str = "upload",
) -> str:
    """Save annotated image to disk, return relative URL path."""
    os.makedirs(evidence_dir, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(evidence_dir, filename)
    cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return f"/evidence/annotated/{filename}"


def create_thumbnail(
    image: np.ndarray,
    max_size: int = 800,
) -> np.ndarray:
    """Resize image to thumbnail while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
