import os
import json
import hashlib
import time
import cv2
import platform
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel

class EvidencePackage(BaseModel):
    violation_id: str
    timestamp: float
    camera_id: str
    violation_type: str
    confidence: float
    license_plate: str
    plate_confidence: float
    raw_frame_path: str
    annotated_frame_path: str
    violation_crop_path: str
    plate_crop_path: Optional[str]
    metadata_path: str
    certificate_path: str
    raw_frame_hash: str
    annotated_frame_hash: str
    metadata_hash: str
    track_id: int
    vehicle_class: str
    vehicle_hash: Optional[str] = None

class EvidenceGenerator:
    """Production-grade evidence package generator for traffic violations."""

    VIOLATION_COLORS = {
        "helmet": (0, 0, 255),          # Red
        "triple_riding": (0, 165, 255),  # Orange
        "seatbelt": (0, 255, 255),       # Yellow
        "wrong_side": (255, 0, 255),     # Magenta
        "red_light": (0, 0, 200),        # Dark Red
        "stop_line": (0, 100, 255),      # Dark Orange
        "illegal_parking": (255, 0, 0),  # Blue
        "speed": (0, 200, 200),          # Teal
    }

    def __init__(self, evidence_dir: str = "./evidence"):
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, raw_frame: np.ndarray, enhanced_frame: np.ndarray, 
                 violation: Dict, tracks: List, camera_config: Dict) -> EvidencePackage:
        """
        Generate complete evidence package:
        - Save raw and annotated frames
        - Save violation crop and license plate crop
        - Compute SHA-256 hashes
        - Generate metadata JSON and BSA Section 63(4)(c) legal certificate
        """
        violation_id = f"VIOL_{int(time.time())}_{violation.get('track_id', 0)}_{uuid.uuid4().hex[:6]}"
        timestamp = time.time()
        camera_id = camera_config.get("camera_id", "CAM_01")
        violation_type = violation.get("type", "unknown")
        confidence = violation.get("confidence", 0.0)
        license_plate = violation.get("license_plate", "UNKNOWN")
        plate_confidence = violation.get("plate_confidence", 0.0)

        # Create folder for this specific violation package
        pkg_dir = self.evidence_dir / violation_id
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # Get crops
        violation_crop = self._crop_object(enhanced_frame, violation.get("bbox"))
        plate_crop = None
        if violation.get("plate_bbox"):
            plate_crop = self._crop_object(enhanced_frame, violation.get("plate_bbox"))

        # Generate annotated frame
        annotated_frame = self.annotate_frame(enhanced_frame, violation, tracks, camera_config)

        # Define file paths
        raw_path = pkg_dir / "raw_frame.jpg"
        annotated_path = pkg_dir / "evidence_annotated.jpg"
        crop_path = pkg_dir / "violation_crop.jpg"
        plate_path = pkg_dir / "plate_crop.jpg" if plate_crop is not None else None
        metadata_path = pkg_dir / "metadata.json"
        cert_path = pkg_dir / "bsa_certificate.txt"

        # Save images
        cv2.imwrite(str(raw_path), raw_frame)
        cv2.imwrite(str(annotated_path), annotated_frame)
        
        if violation_crop is not None and violation_crop.size > 0:
            cv2.imwrite(str(crop_path), violation_crop)
        else:
            # Fallback to saving raw frame copy if crop failed
            cv2.imwrite(str(crop_path), raw_frame)

        if plate_crop is not None and plate_crop.size > 0:
            cv2.imwrite(str(plate_path), plate_crop)

        # Calculate hashes
        raw_hash = self._get_sha256(raw_path)
        annotated_hash = self._get_sha256(annotated_path)

        # Get actual system info instead of hardcoded strings
        sys_node = platform.node() or "Unknown Host"
        sys_arch = platform.machine() or "Unknown Arch"
        device_info = f"TrafficAI Edge Node - {sys_node} ({sys_arch})"
        operating_system = f"{platform.system()} {platform.release()} (v{platform.version()})"

        # Create metadata
        metadata = {
            "violation_id": violation_id,
            "timestamp": timestamp,
            "datetime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
            "camera_id": camera_id,
            "violation_type": violation_type,
            "confidence": float(confidence),
            "license_plate": license_plate,
            "plate_confidence": float(plate_confidence),
            "raw_frame_hash_sha256": raw_hash,
            "annotated_frame_hash_sha256": annotated_hash,
            "device_info": device_info,
            "operating_system": operating_system,
            "vehicle_hash": violation.get("vehicle_hash")
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        metadata_hash = self._get_sha256(metadata_path)

        # Generate legal BSA Section 63(4)(c) Certificate (passing correct metadata_hash)
        self._generate_bsa_certificate(cert_path, metadata, raw_hash, annotated_hash, metadata_hash)

        return EvidencePackage(
            violation_id=violation_id,
            timestamp=timestamp,
            camera_id=camera_id,
            violation_type=violation_type,
            confidence=confidence,
            license_plate=license_plate,
            plate_confidence=plate_confidence,
            raw_frame_path=str(raw_path.relative_to(self.evidence_dir.parent)),
            annotated_frame_path=str(annotated_path.relative_to(self.evidence_dir.parent)),
            violation_crop_path=str(crop_path.relative_to(self.evidence_dir.parent)),
            plate_crop_path=str(plate_path.relative_to(self.evidence_dir.parent)) if plate_path else None,
            metadata_path=str(metadata_path.relative_to(self.evidence_dir.parent)),
            certificate_path=str(cert_path.relative_to(self.evidence_dir.parent)),
            raw_frame_hash=raw_hash,
            annotated_frame_hash=annotated_hash,
            metadata_hash=metadata_hash,
            track_id=int(violation.get("track_id", 0)),
            vehicle_class=violation.get("vehicle_class", "vehicle"),
            vehicle_hash=violation.get("vehicle_hash")
        )

    def annotate_frame(self, frame: np.ndarray, violation: Dict, 
                       tracks: List, camera_config: Dict) -> np.ndarray:
        """Annotate frame with colored bounding boxes, labels, and timestamps."""
        annotated = frame.copy()
        color = self.VIOLATION_COLORS.get(violation.get("type"), (0, 0, 255))
        
        # 1. Draw camera zones if present
        if "stop_line_y" in camera_config:
            y = int(camera_config["stop_line_y"])
            cv2.line(annotated, (0, y), (annotated.shape[1], y), (0, 255, 0), 2)
            cv2.putText(annotated, "STOP LINE", (10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if "no_parking_zones" in camera_config:
            for zone in camera_config["no_parking_zones"]:
                pts = np.array(zone, np.int32)
                cv2.polylines(annotated, [pts], True, (255, 0, 0), 2)
                cv2.putText(annotated, "NO PARKING ZONE", (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 2. Draw bounding box of the violation vehicle/rider
        bbox = violation.get("bbox")
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = f"{violation.get('type').upper()} ({violation.get('confidence', 0.0):.2f})"
            cv2.rectangle(annotated, (x1, y1 - 25), (x1 + len(label)*10 + 20, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 3. Draw license plate annotation if present
        plate_bbox = violation.get("plate_bbox")
        if plate_bbox:
            px1, py1, px2, py2 = map(int, plate_bbox)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
            plate_text = f"PLATE: {violation.get('license_plate', 'UNKNOWN')}"
            cv2.putText(annotated, plate_text, (px1, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Add header status bar / watermark
        hdr_color = (30, 30, 30)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), hdr_color, -1)
        
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        status_text = f"Camera: {camera_config.get('camera_id', 'CAM_01')} | Time: {timestamp_str} | Violation: {violation.get('type').upper()}"
        cv2.putText(annotated, status_text, (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def _crop_object(self, frame: np.ndarray, bbox: Optional[List[float]]) -> Optional[np.ndarray]:
        if not bbox:
            return None
        h, w, _ = frame.shape
        x1, y1, x2, y2 = map(int, bbox)
        
        # Clamp to image boundaries
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 - x1 > 0 and y2 - y1 > 0:
            return frame[y1:y2, x1:x2]
        return None

    def _get_sha256(self, filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _generate_bsa_certificate(self, cert_path: Path, metadata: Dict, 
                                  raw_hash: str, annotated_hash: str, metadata_hash: str) -> None:
        """Generate a certificate under Section 63(4)(c) of the Bharatiya Sakshya Adhiniyam, 2023."""
        cert_text = f"""================================================================================
CERTIFICATE UNDER SECTION 63(4)(c) OF THE BHARATIYA SAKSHYA ADHINIYAM, 2023
(Replaces Section 65B of the Indian Evidence Act, 1872)
================================================================================

I, System Administrator, TrafficAI Control Unit, do hereby certify and declare:

1. I am responsible for the management, operation, and maintenance of the
   Intelligent Transportation System (ITS) node and computer systems operating at:
   Camera ID: {metadata['camera_id']}
   Device Name: {metadata['device_info']}
   Operating System: {metadata['operating_system']}

2. On {metadata['datetime']}, the system recorded a traffic violation of type:
   [{metadata['violation_type'].upper()}] with ID [{metadata['violation_id']}].

3. The electronic record (digital image and metadata) was produced by the
   aforementioned system which operates automatically in the regular course
   of its activities. The system was performing correctly and was in good 
   working order at all times relevant to this record's capture.

4. To ensure electronic record integrity, digital cryptographic hashes have been
   generated at the time of capture:
   - Raw Captured Frame SHA-256 Hash:
     {raw_hash}
   - Annotated Evidence Frame SHA-256 Hash:
     {annotated_hash}
   - Metadata JSON SHA-256 Hash:
     {metadata_hash}

This certificate is signed and submitted in compliance with Section 63(4)(c)
of the Bharatiya Sakshya Adhiniyam, 2023, to establish legal admissibility
of the attached electronic evidence in a court of law.

Date: {metadata['datetime'].split()[0]}
Signature: ____________________________________
(System Administrator, TrafficAI Systems)
"""
        with open(cert_path, "w") as f:
            f.write(cert_text)
