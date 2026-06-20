"""Automatic Number Plate Recognition (ANPR) service with image preprocessing and OCR."""
import re
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List, Any

logger = logging.getLogger(__name__)

PLATE_PATTERNS = {
    "IN": r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}$',
    "US": r'^[A-Z0-9]{5,8}$',
    "UK": r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$',
    "EU": r'^[A-Z]{1,3}\s?\d{1,4}\s?[A-Z]{0,3}$',
}


class ANPRService:
    """End-to-end License Plate Recognition (LPR) pipeline."""

    def __init__(self, ocr_engine: str = "easyocr", region: str = "IN"):
        self.ocr_engine = ocr_engine
        self.region = region
        self.reader = None
        self._init_ocr()

    def _init_ocr(self):
        """Initialize OCR engine."""
        try:
            if self.ocr_engine == "easyocr":
                import easyocr
                # Run OCR on CPU by default (edge deployment friendly), can toggle GPU if available
                self.reader = easyocr.Reader(['en'], gpu=False)
            elif self.ocr_engine == "paddleocr":
                from paddleocr import PaddleOCR
                self.reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except ImportError:
            logger.warning(f"{self.ocr_engine} not installed. ANPR will return empty results.")

    def detect_plate(self, vehicle_bbox: List[float], detections: List[Any]) -> Optional[List[float]]:
        """
        Find the license plate detection from YOLO that lies inside the vehicle bbox.
        Returns the absolute plate bounding box coordinate [x1, y1, x2, y2].
        """
        best_plate = None
        best_iou = 0.0
        
        vx1, vy1, vx2, vy2 = vehicle_bbox
        
        for d in detections:
            # Check if class is license_plate
            cls_name = getattr(d, "class_name", "")
            if not cls_name:
                cls_name = d.get("class_name", "") if isinstance(d, dict) else ""
            if cls_name != "license_plate":
                continue
                
            p_bbox = getattr(d, "bbox", None)
            if not p_bbox:
                p_bbox = d.get("bbox") if isinstance(d, dict) else None
            if not p_bbox:
                continue
                
            px1, py1, px2, py2 = p_bbox
            
            # Check if plate lies inside (or mostly overlaps with) vehicle bbox
            # Intersection of plate with vehicle bbox should be almost 100% of the plate box
            inter_x1 = max(vx1, px1)
            inter_y1 = max(vy1, py1)
            inter_x2 = min(vx2, px2)
            inter_y2 = min(vy2, py2)
            
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            plate_area = (px2 - px1) * (py2 - py1)
            
            containment = inter_area / max(plate_area, 1e-6)
            
            if containment > 0.7:  # If plate is inside the vehicle bbox
                if containment > best_iou:
                    best_iou = containment
                    best_plate = p_bbox
                    
        return best_plate

    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Enhanced license plate preprocessing:
        1. Grayscale conversion
        2. Bilateral filter to denoise while keeping edges sharp
        3. Otsu binarization + morphological operations to close gaps
        4. Perspective deskewing (optional, based on contours)
        5. Resize to 64px height, maintaining aspect ratio
        6. Pad to fixed width (200px)
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        # 1. Grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # 2. Denoise with Bilateral Filter
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 3. Otsu Threshold Binarization
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 4. Resize and pad
        target_h = 64
        h, w = thresh.shape
        aspect = w / h
        target_w = int(target_h * aspect)
        
        resized = cv2.resize(thresh, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Pad or crop to 200px width
        final_w = 200
        if target_w < final_w:
            padding_left = (final_w - target_w) // 2
            padding_right = final_w - target_w - padding_left
            processed = cv2.copyMakeBorder(resized, 0, 0, padding_left, padding_right, 
                                           cv2.BORDER_CONSTANT, value=255) # white pad
        else:
            processed = cv2.resize(resized, (final_w, target_h), interpolation=cv2.INTER_CUBIC)
            
        return processed

    def _deskew_plate(self, img: np.ndarray) -> np.ndarray:
        """Correct minor rotation angles using Hough lines or image moments."""
        coords = np.column_stack(np.where(img > 0))
        angle = 0.0
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
        if abs(angle) < 0.5 or abs(angle) > 20:
            return img
            
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Perform OCR on plate crop with multi-attempt preprocessing pipeline:
        - Attempt 1: Preprocessed Otsu crop
        - Attempt 2: Deskewed version of preprocessed crop
        - Attempt 3: Inverted binary image (for white text on black background)
        Returns (best_text, best_confidence).
        """
        if not self.reader or plate_crop is None or plate_crop.size == 0:
            return ("", 0.0)

        # Build pipeline candidates
        p_otsu = self.preprocess_plate(plate_crop)
        p_deskew = self._deskew_plate(p_otsu)
        p_inverted = cv2.bitwise_not(p_otsu)
        
        candidates = [p_otsu, p_deskew, p_inverted, cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)]
        
        best_text = ""
        best_conf = 0.0

        for img in candidates:
            try:
                if self.ocr_engine == "easyocr":
                    results = self.reader.readtext(img)
                    if results:
                        text = " ".join([r[1] for r in results]).upper().strip()
                        conf = sum(r[2] for r in results) / len(results)
                        
                        # Apply common character corrections (digit / letter confusion)
                        text_corrected = self._post_process_text(text)
                        
                        if conf > best_conf:
                            best_conf = conf
                            best_text = text_corrected
                elif self.ocr_engine == "paddleocr":
                    results = self.reader.ocr(img, cls=True)
                    if results and results[0]:
                        text = " ".join([line[1][0] for line in results[0]]).upper().strip()
                        conf = sum(line[1][1] for line in results[0]) / len(results[0])
                        text_corrected = self._post_process_text(text)
                        
                        if conf > best_conf:
                            best_conf = conf
                            best_text = text_corrected
            except Exception as e:
                logger.debug(f"OCR attempt failed: {e}")

        # Check if validation succeeds
        if not self.validate_plate(best_text):
            # Try validating cleaned version
            cleaned = re.sub(r'[^A-Z0-9]', '', best_text)
            if self.validate_plate(cleaned):
                best_text = cleaned

        return best_text, best_conf

    def _post_process_text(self, text: str) -> str:
        """
        Correct common OCR confusions (e.g. O instead of 0, I instead of 1)
        based on positions in standard license plate formats.
        """
        # Remove special characters/whitespace for standard character checking
        cleaned = re.sub(r'[^A-Z0-9]', '', text)
        
        # Indian Format correction: e.g. MH12AB1234
        # MH (0-1: letters) 12 (2-3: digits) AB (4-5: letters) 1234 (6-9: digits)
        if len(cleaned) >= 8 and len(cleaned) <= 10:
            char_list = list(cleaned)
            
            # Map letter to digit
            to_digit = {"O": "0", "I": "1", "T": "1", "Z": "2", "B": "8", "S": "5", "G": "6"}
            # Map digit to letter
            to_letter = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G"}
            
            # State code must be letters
            for i in [0, 1]:
                if char_list[i] in to_letter:
                    char_list[i] = to_letter[char_list[i]]
                    
            # District code must be digits
            for i in [2, 3]:
                if char_list[i] in to_digit:
                    char_list[i] = to_digit[char_list[i]]
                    
            # Last 4 digits must be numbers
            for i in range(len(char_list) - 4, len(char_list)):
                if char_list[i] in to_digit:
                    char_list[i] = to_digit[char_list[i]]
                    
            return "".join(char_list)

        return cleaned if cleaned else text

    def validate_plate(self, text: str, region: Optional[str] = None) -> bool:
        """Validate plate text against regional regex pattern."""
        r = region or self.region
        pattern = PLATE_PATTERNS.get(r, PLATE_PATTERNS["IN"])
        
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        return bool(re.match(pattern, text.upper().strip())) or bool(re.match(pattern, cleaned))

    def get_vehicle_description(self, vehicle_crop) -> Dict:
        """Fallback: extract vehicle visual description."""
        return {
            "color": "unknown",
            "body_type": "unknown",
            "make": "unknown",
            "model": "unknown",
        }
