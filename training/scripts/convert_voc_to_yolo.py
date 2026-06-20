#!/usr/bin/env python3
"""Convert Pascal VOC XML annotations to YOLO TXT format with train/val splitting."""
import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random

# Default formats
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

def parse_xml(xml_path: Path):
    """Parse XML annotations and return image size and object boxes."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None, []

    size_node = root.find("size")
    if size_node is None:
        return None, []
    
    width_text = size_node.find("width").text
    height_text = size_node.find("height").text
    
    if not width_text or not height_text:
        return None, []
        
    width = int(float(width_text))
    height = int(float(height_text))
    
    if width <= 0 or height <= 0:
        return None, []

    objects = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or not name_node.text:
            continue
        name = name_node.text
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        objects.append({
            "name": name,
            "bbox": (xmin, ymin, xmax, ymax)
        })
    
    return (width, height), objects

def convert_to_yolo_bbox(size, bbox):
    """Convert VOC bbox (xmin, ymin, xmax, ymax) to YOLO normalized (x_center, y_center, w, h)."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Clip coordinates to be inside the image
    x_val = max(0.0, min(1.0, x * dw))
    y_val = max(0.0, min(1.0, y * dh))
    w_val = max(0.0, min(1.0, w * dw))
    h_val = max(0.0, min(1.0, h * dh))
    
    return x_val, y_val, w_val, h_val

def find_image_for_xml(xml_path: Path, search_dir: Path):
    """Find matching image file for an XML file in a search directory."""
    # Check same directory first
    base_name = xml_path.stem
    for ext in IMG_EXTENSIONS:
        img_path = xml_path.parent / (base_name + ext)
        if img_path.exists():
            return img_path
            
    # Search recursively in the search directory if it's different
    if search_dir != xml_path.parent:
        for p in search_dir.rglob(base_name + "*"):
            if p.suffix in IMG_EXTENSIONS:
                return p
    return None

def main():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML to YOLO TXT")
    parser.add_argument("--xml-dir", required=True, help="Directory containing XML files (scanned recursively)")
    parser.add_argument("--img-dir", help="Directory containing Image files (defaults to same as xml-dir)")
    parser.add_argument("--out-dir", required=True, help="Output directory for YOLO format dataset")
    parser.add_argument("--mode", choices=["plate", "helmet", "custom"], default="custom", 
                        help="Pre-configured mapping mode: 'plate' (maps everything to class 0), 'helmet' (maps helmet/head), or 'custom'")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--classes", help="Comma-separated class names for custom mode (e.g. helmet,head,person)")
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    img_dir = Path(args.img_dir) if args.img_dir else xml_dir
    out_dir = Path(args.out_dir)

    # Resolve class names and mappings
    if args.mode == "plate":
        classes = ["license_plate"]
        # Map everything to class ID 0 (since XMLs might contain actual plate strings)
        class_map = lambda name: 0
    elif args.mode == "helmet":
        # We will parse: 'helmet'/'hat' -> 0, 'head'/'hair'/'no_helmet' -> 1
        classes = ["helmet", "no_helmet"]
        def helmet_map(name):
            name_lower = name.lower().strip()
            if "helmet" in name_lower or "hat" in name_lower or "safety" in name_lower:
                return 0
            else:
                return 1
        class_map = helmet_map
    else:
        if not args.classes:
            print("Error: --classes must be specified for 'custom' mode.")
            return
        classes = [c.strip() for c in args.classes.split(",")]
        class_map = lambda name: classes.index(name) if name in classes else -1

    # Create target directories
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Scanning for XML files...")
    all_xmls = list(xml_dir.rglob("*.xml"))
    print(f"Found {len(all_xmls)} XML files.")

    valid_pairs = []
    skipped_no_img = 0
    skipped_invalid_xml = 0

    for xml_path in all_xmls:
        img_path = find_image_for_xml(xml_path, img_dir)
        if not img_path:
            skipped_no_img += 1
            continue
            
        img_size, objects = parse_xml(xml_path)
        if not img_size or not objects:
            skipped_invalid_xml += 1
            continue
            
        valid_pairs.append((xml_path, img_path, img_size, objects))

    print(f"Validated dataset pairs: {len(valid_pairs)}")
    print(f"Skipped (no image found): {skipped_no_img}")
    print(f"Skipped (invalid XML or size <= 0): {skipped_invalid_xml}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * args.split)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"Splitting dataset: {len(train_pairs)} train, {len(val_pairs)} val")

    def process_split(pairs, split_name):
        copied_images = 0
        written_labels = 0
        for xml_path, img_path, img_size, objects in pairs:
            # Output filenames
            out_img_path = out_dir / "images" / split_name / img_path.name
            out_lbl_path = out_dir / "labels" / split_name / (img_path.stem + ".txt")
            
            yolo_lines = []
            for obj in objects:
                cls_id = class_map(obj["name"])
                if cls_id == -1:
                    continue
                yolo_bbox = convert_to_yolo_bbox(img_size, obj["bbox"])
                yolo_lines.append(f"{cls_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
                
            if yolo_lines:
                # Move image to save disk space
                shutil.move(img_path, out_img_path)
                copied_images += 1
                # Write label
                with open(out_lbl_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                written_labels += 1
                
        print(f"  Processed split '{split_name}': moved {copied_images} images, wrote {written_labels} labels.")

    process_split(train_pairs, "train")
    process_split(val_pairs, "val")

    # Generate dataset.yaml
    yaml_content = {
        "path": str(out_dir.resolve().as_posix()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes
    }
    
    import yaml
    with open(out_dir / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"\nSuccessfully converted dataset and saved to {out_dir}")
    print(f"Generated dataset.yaml containing classes: {classes}")

if __name__ == "__main__":
    main()
