#!/usr/bin/env python3
"""Data augmentation pipeline using Albumentations."""
import argparse
import os
import sys
import random
import shutil
import logging
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import albumentations as A
    from albumentations.core.serialization import to_dict
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_augmentation_pipeline():
    """Build augmentation pipeline with bounding box support."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1),
        ], p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.OneOf([
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=1),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=1),
        ], p=0.2),
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
        
        # Simulate degraded CCTV conditions
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
        A.RandomSunFlare(src_radius=200, p=0.15),
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.2),  # Low-res CCTV
        A.ImageCompression(quality_lower=40, quality_upper=90, p=0.3),  # JPEG artifacts
        A.CLAHE(clip_limit=4.0, p=0.2),  # Adaptive histogram
        
        # Night mode: aggressive brightness reduction + heavy noise
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.3), p=1.0),
            A.GaussNoise(var_limit=(30, 80), p=1.0),
        ], p=0.15),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


def read_yolo_labels(label_path: Path):
    """Read YOLO format labels."""
    bboxes = []
    class_labels = []
    if not label_path.exists():
        return bboxes, class_labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                bboxes.append(bbox)
                class_labels.append(cls)
    return bboxes, class_labels


def write_yolo_labels(label_path: Path, bboxes, class_labels):
    """Write YOLO format labels."""
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def augment_dataset(input_dir: Path, output_dir: Path, multiplier: int = 3, preview: bool = False):
    """Apply augmentations to expand dataset."""
    if not HAS_CV2:
        print("Error: OpenCV not installed. Run: pip install opencv-python")
        sys.exit(1)
    if not HAS_ALBUM:
        print("Error: albumentations not installed. Run: pip install albumentations")
        sys.exit(1)
    
    transform = build_augmentation_pipeline()
    img_dir = input_dir / "images"
    lbl_dir = input_dir / "labels"
    
    if not img_dir.exists():
        for split in ["train", "val"]:
            if (input_dir / "images" / split).exists():
                img_dir = input_dir / "images" / split
                lbl_dir = input_dir / "labels" / split
                break
    
    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    if preview:
        preview_dir = output_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not image_files:
        logger.error(f"No images found in {img_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images")
    logger.info(f"Augmenting with multiplier={multiplier}")
    
    total_augmented = 0
    preview_count = 0
    
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        bboxes, class_labels = read_yolo_labels(lbl_path)
        
        # Copy original
        shutil.copy2(img_path, out_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)
        
        for aug_idx in range(multiplier):
            try:
                augmented = transform(
                    image=image,
                    bboxes=bboxes if bboxes else [],
                    class_labels=class_labels if class_labels else [],
                )
                
                aug_image = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["class_labels"]
                
                aug_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
                aug_img_rgb = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_img_dir / aug_name), aug_img_rgb)
                
                aug_lbl_name = f"{img_path.stem}_aug{aug_idx}.txt"
                write_yolo_labels(out_lbl_dir / aug_lbl_name, aug_bboxes, aug_labels)
                
                total_augmented += 1
                
                if preview and preview_count < 5:
                    cv2.imwrite(str(preview_dir / f"preview_{preview_count}.jpg"), aug_img_rgb)
                    preview_count += 1
                    
            except Exception as e:
                logger.warning(f"Augmentation failed for {img_path.name} (iter {aug_idx}): {e}")
    
    logger.info(f"\nAugmentation complete!")
    logger.info(f"  Original images: {len(image_files)}")
    logger.info(f"  Augmented images: {total_augmented}")
    logger.info(f"  Total images: {len(image_files) + total_augmented}")


def main():
    parser = argparse.ArgumentParser(description="Augment training dataset")
    parser.add_argument("--input-dir", required=True, help="Input dataset directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for augmented data")
    parser.add_argument("--multiplier", type=int, default=3, help="Augmentation multiplier per image")
    parser.add_argument("--preview", action="store_true", help="Save preview samples")
    args = parser.parse_args()
    
    augment_dataset(Path(args.input_dir), Path(args.output_dir), args.multiplier, args.preview)


if __name__ == "__main__":
    main()
