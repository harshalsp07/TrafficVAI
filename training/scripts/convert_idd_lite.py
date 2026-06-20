#!/usr/bin/env python3
"""Convert IDD Lite (idd20k_lite) segmentation masks to YOLO bounding box format."""
import os
import shutil
import cv2
import numpy as np
from pathlib import Path

# IDD Lite classes of interest for bounding box extraction:
# 2: Living things (pedestrian/rider) -> Maps to YOLO class 0: living_thing
# 3: Vehicles -> Maps to YOLO class 1: vehicle
CLASS_MAPPING = {
    2: 0, # living_thing
    3: 1  # vehicle
}

CLASSES = ["living_thing", "vehicle"]

def extract_boxes_from_mask(mask_path: Path):
    """Find connected components for each class and return bounding boxes."""
    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        return []

    height, width = mask.shape
    boxes = []

    for src_cls, dst_cls in CLASS_MAPPING.items():
        # Create a binary mask for the specific class
        class_mask = (mask == src_cls).astype(np.uint8) * 255
        
        # Find connected components (individual objects)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask)
        
        for i in range(1, num_labels): # 0 is background
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out tiny noise regions
            if area < 15:
                continue
                
            # Convert to YOLO format
            x_center = (x + w / 2.0) / width
            y_center = (y + h / 2.0) / height
            norm_w = w / width
            norm_h = h / height
            
            boxes.append((dst_cls, x_center, y_center, norm_w, norm_h))
            
    return boxes

def main():
    base_dir = Path("training/datasets/idd20k_lite")
    out_dir = Path("training/datasets/idd_lite_detection")

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        return

    print("Converting IDD Lite segmentation masks to YOLO bounding boxes...")

    # Create target directories
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        img_root = base_dir / "leftImg8bit" / split
        lbl_root = base_dir / "gtFine" / split

        if not img_root.exists():
            continue

        print(f"Processing split: {split}")
        copied_count = 0
        labeled_count = 0

        # Scan for images in subfolders
        image_files = list(img_root.rglob("*_image.jpg"))
        for img_path in image_files:
            # Find matching label image
            # Image:  leftImg8bit/train/0/frame0000_image.jpg
            # Label:  gtFine/train/0/frame0000_label.png
            relative_parts = img_path.relative_to(img_root)
            seq_dir = relative_parts.parent
            base_name = img_path.name.replace("_image.jpg", "")
            
            lbl_path = lbl_root / seq_dir / f"{base_name}_label.png"
            
            # Test split does not have labels in gtFine
            boxes = []
            if lbl_path.exists():
                boxes = extract_boxes_from_mask(lbl_path)
            
            # Save if test split or if we found valid boxes
            if split == "test" or len(boxes) > 0:
                # Output paths
                out_img_path = out_dir / "images" / split / f"{base_name}.jpg"
                out_lbl_path = out_dir / "labels" / split / f"{base_name}.txt"

                # Move image to save disk space
                shutil.move(str(img_path), str(out_img_path))
                copied_count += 1

                if split != "test":
                    with open(out_lbl_path, "w") as f:
                        lines = [f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in boxes]
                        f.write("\n".join(lines))
                    labeled_count += 1

        print(f"  Split '{split}': Moved {copied_count} images, wrote {labeled_count} labels.")

    # Clean up the original extracted folder to free up space
    shutil.rmtree(base_dir)
    print(f"Removed temporary directory {base_dir}")

    # Generate dataset.yaml
    yaml_content = f"""# IDD Lite Object Detection Dataset
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(out_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nSuccessfully converted IDD Lite. Saved to {out_dir}")

if __name__ == "__main__":
    main()
