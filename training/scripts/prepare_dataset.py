#!/usr/bin/env python3
"""Dataset preparation utility with synthetic data generation."""
import argparse
import json
import os
import random
import shutil
import sys
import yaml
from pathlib import Path

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

CLASSES = ["car", "truck", "motorcycle", "bus", "auto_rickshaw", "bicycle", "pedestrian"]
COLORS = [
    (180, 180, 180), (100, 100, 100), (200, 50, 50), (50, 50, 200),
    (50, 200, 50), (200, 200, 50), (200, 100, 50), (100, 50, 200),
]


def generate_synthetic_dataset(output_dir: Path, num_images: int = 50):
    """Generate synthetic images with bounding box labels."""
    if not HAS_PIL:
        print("Error: Pillow not installed. Run: pip install Pillow")
        sys.exit(1)
    
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    splits = {"train": int(num_images * 0.8), "val": int(num_images * 0.1), "test": num_images - int(num_images * 0.8) - int(num_images * 0.1)}
    if splits["test"] <= 0:
        splits["test"] = max(1, num_images - splits["train"] - splits["val"])
    
    img_idx = 0
    for split, count in splits.items():
        for i in range(count):
            w, h = 640, 480
            bg_color = (random.randint(30, 80), random.randint(30, 80), random.randint(30, 80))
            img = Image.new("RGB", (w, h), bg_color)
            draw = ImageDraw.Draw(img)
            
            # Draw road
            draw.rectangle([0, h//2 - 60, w, h//2 + 60], fill=(60, 60, 60))
            draw.line([(0, h//2), (w, h//2)], fill=(255, 255, 255), width=2)
            
            labels = []
            num_objects = random.randint(2, 6)
            for _ in range(num_objects):
                cls_id = random.randint(0, len(CLASSES) - 1)
                color = random.choice(COLORS)
                
                obj_w = random.randint(40, 120)
                obj_h = random.randint(30, 90)
                x1 = random.randint(10, w - obj_w - 10)
                y1 = random.randint(h//2 - 80, h//2 + 40)
                x2 = x1 + obj_w
                y2 = y1 + obj_h
                
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255))
                if cls_id <= 3:
                    wheel_y = y2 - 8
                    draw.ellipse([x1 + 5, wheel_y, x1 + 15, y2], fill=(30, 30, 30))
                    draw.ellipse([x2 - 15, wheel_y, x2 - 5, y2], fill=(30, 30, 30))
                
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = obj_w / w
                bh = obj_h / h
                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
            img_name = f"img_{img_idx:04d}.jpg"
            img.save(output_dir / "images" / split / img_name)
            
            lbl_name = f"img_{img_idx:04d}.txt"
            with open(output_dir / "labels" / split / lbl_name, "w") as f:
                f.write("\n".join(labels))
            
            img_idx += 1
    
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"\nSynthetic dataset created: {output_dir}")
    print(f"  Train: {splits['train']} images")
    print(f"  Val:   {splits['val']} images")
    print(f"  Test:  {splits['test']} images")
    print(f"  Classes: {len(CLASSES)}")


def convert_coco_to_yolo(coco_json: str, output_dir: Path):
    """Convert COCO JSON annotations to YOLO TXT format."""
    with open(coco_json) as f:
        coco = json.load(f)
    
    images = {img["id"]: img for img in coco["images"]}
    cat_map = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}
    
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    count = 0
    for img_id, anns in annotations_by_image.items():
        img_info = images[img_id]
        w, h = img_info["width"], img_info["height"]
        filename = Path(img_info["file_name"]).stem + ".txt"
        
        lines = []
        for ann in anns:
            cls_id = cat_map.get(ann["category_id"], 0)
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        with open(output_dir / "labels" / filename, "w") as f:
            f.write("\n".join(lines))
        count += 1
    
    print(f"Converted {count} images from COCO to YOLO format")


def verify_dataset(dataset_dir: Path):
    """Verify dataset integrity."""
    print(f"\nVerifying dataset: {dataset_dir}")
    issues = 0
    total_images = 0
    class_counts = {}
    
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists():
            continue
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        total_images += len(images)
        
        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                print(f"  Warning: No label for {img_path.name}")
                issues += 1
                continue
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
    
    print(f"  Total images: {total_images}")
    print(f"  Issues: {issues}")
    print(f"  Class distribution:")
    for cls_id, count in sorted(class_counts.items()):
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
        print(f"    {name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--output", type=str, default="./datasets/sample", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dataset")
    parser.add_argument("--num-images", type=int, default=50, help="Number of synthetic images")
    parser.add_argument("--coco-json", type=str, help="Convert COCO JSON to YOLO format")
    parser.add_argument("--verify", action="store_true", help="Verify dataset integrity")
    parser.add_argument("--split-ratio", type=str, default="0.8,0.1,0.1", help="Train/val/test split ratio")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.synthetic:
        generate_synthetic_dataset(output_dir, args.num_images)
    
    if args.coco_json:
        convert_coco_to_yolo(args.coco_json, output_dir)
    
    if args.verify or args.synthetic:
        verify_dataset(output_dir)
    
    if not args.synthetic and not args.coco_json and not args.verify:
        print("Specify --synthetic, --coco-json, or --verify")
        parser.print_help()


if __name__ == "__main__":
    main()
