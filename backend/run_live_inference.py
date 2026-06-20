import os
import sys
import cv2
import time
import sqlite3
import argparse
from datetime import datetime

# Add app to path to import components if running from backend folder
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Now import the UnifiedPipeline and related classes
try:
    from app.services.unified_pipeline import UnifiedPipeline
    from app.config import settings
    HAS_PIPELINE = True
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    HAS_PIPELINE = False

def run_pipeline(source, weights_path, camera_id, device, headless=False, max_frames=None):
    if not HAS_PIPELINE:
        print("Error: Could not import UnifiedPipeline service. Exit.")
        return

    # Override config paths from CLI args
    settings.MODEL_PATH = weights_path
    settings.DEVICE = device
    settings.ensure_directories()

    db_path = settings.sqlite_path
    
    print("\n" + "="*60)
    print("TrafficAI Live Real-Data Cascade Pipeline")
    print("="*60)
    print(f"Video Source: {source}")
    print(f"Model Weights: {weights_path}")
    print(f"Device: {device}")
    print(f"Target Camera DB ID: {camera_id}")
    print(f"SQLite Database: {db_path}")
    print("="*60 + "\n")

    # Initialize the Unified Pipeline
    pipeline = UnifiedPipeline()

    # Define camera spatial configuration (default Indian lane rules)
    # Zone coords for illegal parking (bounding box)
    no_parking_zone = [
        [100.0, 300.0],
        [300.0, 300.0],
        [300.0, 480.0],
        [100.0, 480.0]
    ]

    camera_config = {
        "camera_id": f"CAM_{camera_id:02d}",
        "stop_line_y": 350.0,
        "signal_state": "red",  # set red to allow stop-line and red-light checks
        "lane_direction": (0.0, 1.0), # moving down
        "no_parking_zones": [no_parking_zone],
        "speed_limit": settings.SPEED_LIMIT_KMH,
        "sahi_enabled": settings.SAHI_ENABLED
    }

    # Connect to SQLite DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Try opening video stream
    is_live = False
    if str(source).isdigit():
        source = int(source)
        is_live = True
    elif str(source).startswith("rtsp://") or str(source).startswith("http://"):
        is_live = True
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        conn.close()
        return

    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video stream finished or disconnected.")
                break

            frame_count += 1
            
            # Periodically toggle traffic signal state to test red-light/green-light dynamics
            if frame_count % 150 == 0:
                camera_config["signal_state"] = "green" if camera_config["signal_state"] == "red" else "red"
                print(f"[SYSTEM] Junction Signal Changed to: {camera_config['signal_state'].upper()}")

            # Process frame through the unified pipeline
            result = pipeline.process_frame(frame, camera_config)

            # Log and persist any violations detected
            for ev in result.evidence_packages:
                print(f"[VIOLATION LOGGED] ID: {ev.violation_id} | Type: {ev.violation_type.upper()} | "
                      f"Class: {ev.vehicle_class} | Plate: {ev.license_plate} (Conf: {ev.plate_confidence:.2f})")
                
                # Insert into DB
                cursor.execute(
                    """INSERT OR REPLACE INTO traffic_violations 
                       (violation_id, violation_time, camera_id, track_id, vehicle_class, license_plate, plate_confidence, violation_type, confidence, evidence_image_path, evidence_crop_path, sha256_hash, status, vehicle_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ev.violation_id,
                        datetime.fromtimestamp(ev.timestamp).isoformat(),
                        camera_id,
                        ev.track_id,
                        ev.vehicle_class, # mapping class type correctly
                        ev.license_plate,
                        float(ev.plate_confidence),
                        ev.violation_type,
                        float(ev.confidence),
                        ev.annotated_frame_path,
                        ev.violation_crop_path,
                        ev.annotated_frame_hash,
                        "pending",
                        ev.vehicle_hash
                    )
                )
                conn.commit()

            # Visual Feedback Window
            # Draw tracks and active zones on screen
            for track in result.tracks:
                bbox = [int(x) for x in track.bbox]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                label = f"ID {track.track_id}: {track.class_name} ({track.state})"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw stop line
            sy = int(camera_config["stop_line_y"])
            line_color = (0, 0, 255) if camera_config["signal_state"] == "red" else (0, 255, 0)
            cv2.line(frame, (0, sy), (frame.shape[1], sy), line_color, 2)
            cv2.putText(frame, f"STOP LINE ({camera_config['signal_state'].upper()})", (15, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)

            # Show window
            if not headless:
                cv2.imshow("TrafficAI — Unified Cascade inference", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
            
            if max_frames and frame_count >= max_frames:
                print(f"Reached max frames ({max_frames}). Exiting...")
                break

            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Processed {frame_count} frames | Current FPS: {fps:.1f} | Preprocess Quality: {result.quality.dict()}")

    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        conn.close()
        print("Pipeline shut down.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live video through TrafficAI Unified Cascade Pipeline")
    parser.add_argument("--source", default="0", help="Path to video file or camera index (default: 0)")
    parser.add_argument("--weights", default="./weights/yolov11m.pt", help="Path to YOLO model weights")
    parser.add_argument("--camera-id", type=int, default=1, help="Camera database ID to associate violations with")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing N frames")
    args = parser.parse_args()
    
    run_pipeline(
        args.source, 
        args.weights, 
        args.camera_id, 
        args.device, 
        headless=args.headless, 
        max_frames=args.max_frames
    )
