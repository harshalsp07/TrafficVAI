#!/usr/bin/env python3
"""Model evaluation and benchmarking script."""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def evaluate(weights: str, data: str, imgsz: int = 640, device: str = "cpu", batch: int = 16, conf: float = 0.25):
    """Run model validation on dataset."""
    model = YOLO(weights)
    print(f"\n{'='*60}")
    print(f"Evaluating: {weights}")
    print(f"Dataset: {data}")
    print(f"Device: {device}, Image size: {imgsz}")
    print(f"{'='*60}\n")
    
    results = model.val(data=data, imgsz=imgsz, device=device, batch=batch, conf=conf)
    
    metrics = {
        "mAP50": float(getattr(results, 'map50', 0) if hasattr(results, 'map50') else 0),
        "mAP50_95": float(getattr(results, 'map', 0) if hasattr(results, 'map') else 0),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)) if hasattr(results, 'results_dict') else 0,
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)) if hasattr(results, 'results_dict') else 0,
    }
    
    p = metrics["precision"]
    r = metrics["recall"]
    metrics["f1"] = round(2 * p * r / max(p + r, 1e-6), 4)
    
    print(f"\nResults:")
    print(f"  mAP@50:      {metrics['mAP50']:.4f}")
    print(f"  mAP@50:95:   {metrics['mAP50_95']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    
    return metrics, results


def benchmark_speed(weights: str, imgsz: int = 640, device: str = "cpu", n_frames: int = 100):
    """Benchmark inference speed."""
    import numpy as np
    model = YOLO(weights)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(5):
        model(dummy, verbose=False, device=device)
    
    # Benchmark
    start = time.time()
    for _ in range(n_frames):
        model(dummy, verbose=False, device=device)
    elapsed = time.time() - start
    
    avg_ms = (elapsed / n_frames) * 1000
    fps = n_frames / elapsed
    
    print(f"\nSpeed Benchmark ({n_frames} frames):")
    print(f"  Avg latency: {avg_ms:.2f} ms/frame")
    print(f"  Throughput:  {fps:.1f} FPS")
    
    return {"avg_latency_ms": round(avg_ms, 2), "fps": round(fps, 1)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate and benchmark YOLO model")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--data", help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    parser.add_argument("--n-frames", type=int, default=100, help="Frames for benchmark")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()
    
    if not HAS_ULTRALYTICS:
        print("Error: ultralytics not installed")
        sys.exit(1)
    
    results_data = {"weights": args.weights, "timestamp": datetime.now().isoformat()}
    
    if args.data:
        metrics, _ = evaluate(args.weights, args.data, args.imgsz, args.device, args.batch, args.conf_threshold)
        results_data["metrics"] = metrics
    
    if args.benchmark:
        speed = benchmark_speed(args.weights, args.imgsz, args.device, args.n_frames)
        results_data["speed"] = speed
    
    if not args.data and not args.benchmark:
        print("Specify --data for evaluation or --benchmark for speed test")
        sys.exit(1)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
