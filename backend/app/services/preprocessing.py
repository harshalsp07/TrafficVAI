import cv2
import numpy as np
from pydantic import BaseModel

class FrameQuality(BaseModel):
    blur_score: float
    brightness: float
    noise_level: float
    weather_score: float  # Entropy or contrast indicator
    is_dark: bool
    is_blurry: bool
    is_noisy: bool
    is_hazy: bool

class FramePreprocessor:
    """Adaptive frame quality enhancement pipeline for real-time ITS."""

    def __init__(self, blur_threshold: float = 80.0, dark_threshold: float = 65.0, 
                 noise_threshold: float = 15.0, haze_threshold: float = 40.0):
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.noise_threshold = noise_threshold
        self.haze_threshold = haze_threshold

    def assess_quality(self, frame: np.ndarray) -> FrameQuality:
        """Compute metrics to assess image quality."""
        # 1. Grayscale version for fast analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Blur score using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 3. Brightness using mean value of the histogram
        brightness = float(np.mean(gray))
        
        # 4. Noise level using high-frequency energy in Laplacian
        # An alternative is computing the standard deviation of local variances
        noise_level = float(np.std(cv2.absdiff(gray, cv2.GaussianBlur(gray, (3, 3), 0))))
        
        # 5. Weather score using entropy/contrast (std dev of grayscale histogram)
        weather_score = float(np.std(gray))
        
        is_dark = brightness < self.dark_threshold
        is_blurry = blur_score < self.blur_threshold
        is_noisy = noise_level > self.noise_threshold
        is_hazy = weather_score < self.haze_threshold and not is_dark  # low contrast but not dark implies haze/fog
        
        return FrameQuality(
            blur_score=blur_score,
            brightness=brightness,
            noise_level=noise_level,
            weather_score=weather_score,
            is_dark=is_dark,
            is_blurry=is_blurry,
            is_noisy=is_noisy,
            is_hazy=is_hazy
        )

    def enhance(self, frame: np.ndarray, quality: FrameQuality) -> np.ndarray:
        """Apply necessary corrections based on quality assessment."""
        enhanced = frame.copy()
        
        # If the frame quality is fine, return a copy immediately (zero/negligible overhead)
        if not (quality.is_dark or quality.is_hazy or quality.is_blurry or quality.is_noisy):
            return enhanced

        # 1. Low light enhancement using CLAHE in LAB space (applies to L channel)
        if quality.is_dark:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            merged = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
        # 2. Fog/haze correction using a simplified Dark Channel Prior or contrast adjustment
        if quality.is_hazy:
            # Simplified DCP: subtract dark channel to boost contrast locally
            # We will use local CLAHE and unsharp masking to clear haze
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            cl = clahe.apply(l_channel)
            merged = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
            # Additional sharpening for dehazing effect
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        # 3. Noise correction using fast bilateral filtering
        if quality.is_noisy:
            # Bilateral filter preserves edges (important for license plates and helmets)
            enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)

        # 4. Motion blur correction using Laplacian sharpening
        if quality.is_blurry:
            # Simple sharpening kernel to reverse mild blur
            kernel = np.array([[0, -1, 0], 
                               [-1, 5, -1], 
                               [0, -1, 0]], dtype=np.float32)
            enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced
