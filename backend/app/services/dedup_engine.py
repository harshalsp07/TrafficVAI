"""Deduplication Engine combining Perceptual Hashing (dHash) and Database-level guards."""

import cv2
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.config import settings

logger = logging.getLogger(__name__)

def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for an image.
    
    Fast and robust to resizing, brightness, and contrast changes.
    """
    if image is None or image.size == 0:
        return ""
    try:
        # Convert to grayscale and resize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize to (hash_size + 1, hash_size)
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        # Compute difference
        diff = resized[:, 1:] > resized[:, :-1]
        # Convert binary array to hex string
        decimal_value = 0
        for row in diff:
            for val in row:
                decimal_value = (decimal_value << 1) | int(val)
        # Format as 16-character hex string (for 8x8 hash)
        return f"{decimal_value:016x}"
    except Exception as e:
        logger.error(f"Error computing dHash: {e}")
        return ""

def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hashes."""
    if not hash1 or not hash2:
        return 999
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        # XOR and count set bits
        return bin(val1 ^ val2).count('1')
    except ValueError:
        return 999


class DedupEngine:
    """4th and 5th Layer Deduplication Engine."""

    def __init__(self):
        # In-memory store of recent hashes: (camera_id, violation_type) -> list of (timestamp, hash)
        self._hash_history: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}
        self.last_cleanup = time.time()

    def should_record(self, camera_id: str, violation_type: str, 
                      vehicle_crop: Optional[np.ndarray]) -> Tuple[bool, str]:
        """Determine if a violation should be recorded based on perceptual hash similarity.
        
        Returns:
            (should_record: bool, hash_str: str)
        """
        now = time.time()
        self._periodic_cleanup(now)

        if vehicle_crop is None or vehicle_crop.size == 0:
            return True, ""

        # Compute hash
        hash_str = compute_dhash(vehicle_crop)
        if not hash_str:
            return True, ""

        key = (camera_id, violation_type)
        history = self._hash_history.get(key, [])

        # Check in-memory hash history for matches within window
        window_sec = getattr(settings, "DEDUP_HASH_WINDOW_MINUTES", 5) * 60
        threshold = getattr(settings, "DEDUP_HASH_THRESHOLD", 10)

        for ts, prev_hash in history:
            if now - ts <= window_sec:
                dist = hamming_distance(hash_str, prev_hash)
                if dist < threshold:
                    logger.info(
                        f"[DEDUPLICATION] Suppressed duplicate violation of type '{violation_type}' "
                        f"on camera '{camera_id}' due to perceptual hash similarity (distance: {dist})"
                    )
                    return False, hash_str

        # Add to history
        if key not in self._hash_history:
            self._hash_history[key] = []
        self._hash_history[key].append((now, hash_str))

        return True, hash_str

    def _periodic_cleanup(self, now: float) -> None:
        """Purge entries older than the hash window from memory."""
        if now - self.last_cleanup < 60.0:
            return

        window_sec = getattr(settings, "DEDUP_HASH_WINDOW_MINUTES", 5) * 60
        new_history = {}
        for key, items in self._hash_history.items():
            valid_items = [item for item in items if now - item[0] <= window_sec]
            if valid_items:
                new_history[key] = valid_items

        self._hash_history = new_history
        self.last_cleanup = now
