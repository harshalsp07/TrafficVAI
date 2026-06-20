#!/usr/bin/env python3
"""Export trained YOLO models to various formats."""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


SUPPORTED_FORMATS = ["onnx", "openvino", "engine", "torchscript", "coreml"]


def export_model(weights: str, fmt: str, imgsz: int = 640, half: bool = False, int8: bool = False, device: str = "cpu"):
    """Export model to specified format."""
    print(f"\n{'='*60}")
    print(f"Exporting: {weights} -> {fmt}")
    print(f"{'='*60}")
    
    model = YOLO(weights)
    export_args = {"format": fmt, "imgsz": imgsz}
    if half:
        export_args["half"] = True
    if int8:
        export_args["int8"] = True
    if fmt == "engine":
        export_args["device"] = device
    
    export_path = model.export(**export_args)
    
    if export_path:
        p = Path(export_path) if isinstance(export_path, str) else Path(str(export_path))
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"\nExported: {p}")
            print(f"Size: {size_mb:.2f} MB")
        else:
            print(f"\nExport reported path: {export_path}")
    return export_path


def validate_export(export_path: str):
    """Validate export by running test inference."""
    try:
        import numpy as np
        model = YOLO(export_path)
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy, verbose=False)
        print(f"Validation OK: {len(results)} result(s)")
        return True
    except Exception as e:
        print(f"Validation warning: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to deployment formats")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--format", choices=SUPPORTED_FORMATS, help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", help="FP16 half precision")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser.add_argument("--device", default="cpu", help="Device for TensorRT export")
    parser.add_argument("--all", action="store_true", help="Export to all formats")
    parser.add_argument("--validate", action="store_true", help="Validate export")
    args = parser.parse_args()
    
    if not HAS_ULTRALYTICS:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    if not Path(args.weights).exists():
        print(f"Error: Weights not found: {args.weights}")
        sys.exit(1)
    
    formats = SUPPORTED_FORMATS if args.all else ([args.format] if args.format else ["onnx"])
    
    for fmt in formats:
        try:
            path = export_model(args.weights, fmt, args.imgsz, args.half, args.int8, args.device)
            if args.validate and path:
                validate_export(str(path))
        except Exception as e:
            print(f"Export to {fmt} failed: {e}")
    
    print(f"\nExport complete!")


if __name__ == "__main__":
    main()
