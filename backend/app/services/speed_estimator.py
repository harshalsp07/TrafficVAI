"""Speed estimation via ground-plane homography."""
import math
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class SpeedEstimator:
    """Homography-based speed estimation."""

    def __init__(self):
        self.homography_matrix = None
        self.is_calibrated = False

    def set_homography(self, src_points: List[List[float]], dst_points: List[List[float]]) -> bool:
        """Compute homography from pixel to world coordinates."""
        try:
            import numpy as np
            import cv2
            src = np.float32(src_points)
            dst = np.float32(dst_points)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                self.homography_matrix = H
                self.is_calibrated = True
                logger.info("Homography matrix computed successfully")
                return True
            return False
        except ImportError:
            logger.warning("OpenCV not available for homography computation")
            return False
        except Exception as e:
            logger.error(f"Homography computation failed: {e}")
            return False

    def transform_point(self, pixel_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Transform pixel coordinates to world coordinates."""
        if not self.is_calibrated:
            return None
        try:
            import numpy as np
            pt = np.array([pixel_point[0], pixel_point[1], 1.0])
            world = self.homography_matrix @ pt
            world = world / world[2]
            return (float(world[0]), float(world[1]))
        except Exception:
            return None

    def estimate_speed(
        self,
        track_history: List[Tuple[float, float]],
        fps: float = 30.0,
        window: int = 5,
    ) -> float:
        """Estimate speed in km/h from track history."""
        if len(track_history) < 2:
            return 0.0

        points = track_history[-window:] if len(track_history) >= window else track_history

        if self.is_calibrated:
            world_points = []
            for p in points:
                wp = self.transform_point(p)
                if wp:
                    world_points.append(wp)
            if len(world_points) >= 2:
                total_dist = 0.0
                for i in range(1, len(world_points)):
                    dx = world_points[i][0] - world_points[i-1][0]
                    dy = world_points[i][1] - world_points[i-1][1]
                    total_dist += math.sqrt(dx*dx + dy*dy)
                time_elapsed = (len(world_points) - 1) / max(fps, 1.0)
                speed_ms = total_dist / max(time_elapsed, 0.001)
                speed_kmh = speed_ms * 3.6
                return round(speed_kmh, 1)

        total_pixel_dist = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            total_pixel_dist += math.sqrt(dx*dx + dy*dy)
        time_elapsed = (len(points) - 1) / max(fps, 1.0)
        pixel_speed = total_pixel_dist / max(time_elapsed, 0.001)
        estimated_kmh = pixel_speed * 0.1
        return round(estimated_kmh, 1)
