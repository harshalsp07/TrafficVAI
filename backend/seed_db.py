import os
import sqlite3
import uuid
import random
from datetime import datetime, timedelta

# Create database directory if it doesn't exist
os.makedirs("./data", exist_ok=True)
db_path = "./data/traffic.db"

# Schema SQL
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cameras (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    rtsp_url        TEXT    NOT NULL,
    location_lat    REAL,
    location_lon    REAL,
    status          TEXT    NOT NULL DEFAULT 'inactive',
    zone_config     TEXT,               -- JSON blob
    calibration_data TEXT,              -- JSON blob
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS vehicle_trajectories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_time  TEXT    NOT NULL,
    camera_id       INTEGER NOT NULL REFERENCES cameras(id),
    track_id        INTEGER NOT NULL,
    vehicle_class   TEXT,
    centroid_x      REAL    NOT NULL,
    centroid_y      REAL    NOT NULL,
    speed           REAL,
    lane_id         TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS traffic_violations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id        TEXT    NOT NULL UNIQUE,       -- UUID v4
    violation_time      TEXT    NOT NULL,
    camera_id           INTEGER NOT NULL REFERENCES cameras(id),
    track_id            INTEGER,
    vehicle_class       TEXT,
    license_plate       TEXT,
    plate_confidence    REAL,
    violation_type      TEXT    NOT NULL,
    confidence          REAL    NOT NULL,
    evidence_image_path TEXT,
    evidence_crop_path  TEXT,
    sha256_hash         TEXT,
    status              TEXT    NOT NULL DEFAULT 'pending',
    vehicle_hash        TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,
    model_name      TEXT    NOT NULL,
    dataset         TEXT    NOT NULL,
    epochs          INTEGER NOT NULL,
    batch_size      INTEGER NOT NULL DEFAULT 16,
    status          TEXT    NOT NULL DEFAULT 'queued',
    current_epoch   INTEGER NOT NULL DEFAULT 0,
    best_map        REAL,
    loss            REAL,
    started_at      TEXT,
    completed_at    TEXT,
    config          TEXT,               -- JSON blob
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS traffic_density_hourly (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    window_hour     TEXT    NOT NULL,
    camera_id       INTEGER NOT NULL REFERENCES cameras(id),
    lane_id         TEXT,
    total_vehicles  INTEGER NOT NULL DEFAULT 0,
    average_speed   REAL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_violations_type ON traffic_violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_violations_camera ON traffic_violations(camera_id);
CREATE INDEX IF NOT EXISTS idx_violations_time ON traffic_violations(violation_time);
CREATE INDEX IF NOT EXISTS idx_violations_status ON traffic_violations(status);
CREATE INDEX IF NOT EXISTS idx_violations_vehicle_hash ON traffic_violations(vehicle_hash);
CREATE INDEX IF NOT EXISTS idx_trajectories_camera ON vehicle_trajectories(camera_id);
CREATE INDEX IF NOT EXISTS idx_density_camera_hour ON traffic_density_hourly(camera_id, window_hour);
"""

def seed_db():
    if os.path.exists(db_path):
        print(f"Removing old database at {db_path}...")
        try:
            os.remove(db_path)
        except Exception as e:
            print(f"Could not remove database file: {e}")
    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Creating tables and indexes...")
    cursor.executescript(SCHEMA_SQL)
    conn.commit()

    # Clear existing data to avoid duplicates
    print("Clearing old data...")
    cursor.execute("DELETE FROM traffic_density_hourly")
    cursor.execute("DELETE FROM training_runs")
    cursor.execute("DELETE FROM traffic_violations")
    cursor.execute("DELETE FROM vehicle_trajectories")
    cursor.execute("DELETE FROM cameras")
    conn.commit()

    # 1. Seed Cameras
    cameras_data = [
        ('Cam-01: Silk Board Junction', 'rtsp://192.168.1.101/stream1', 12.9176, 77.6244, 'online'),
        ('Cam-02: Electronic City Toll', 'rtsp://192.168.1.102/stream1', 12.8499, 77.6663, 'online'),
        ('Cam-03: Indiranagar 100ft Rd', 'rtsp://192.168.1.103/stream1', 12.9718, 77.6412, 'online'),
        ('Cam-04: Marathahalli Bridge', 'rtsp://192.168.1.104/stream1', 12.9592, 77.6974, 'offline'),
        ('Cam-05: Hebbal Flyover', 'rtsp://192.168.1.105/stream1', 13.0354, 77.5988, 'online'),
        ('Cam-06: Majestic Circle', 'rtsp://192.168.1.106/stream1', 12.9766, 77.5729, 'online'),
        ('Cam-07: MG Road Junction', 'rtsp://192.168.1.107/stream1', 12.9738, 77.6119, 'online'),
        ('Cam-08: Yeshwanthpur Chowk', 'rtsp://192.168.1.108/stream1', 13.0285, 77.5402, 'offline')
    ]
    
    print("Seeding cameras...")
    camera_ids = []
    for name, rtsp, lat, lon, status in cameras_data:
        cursor.execute(
            "INSERT INTO cameras (name, rtsp_url, location_lat, location_lon, status) VALUES (?, ?, ?, ?, ?)",
            (name, rtsp, lat, lon, status)
        )
        camera_ids.append(cursor.lastrowid)
    conn.commit()

    # 2. Seed Violations
    violation_types = ['helmet', 'speed', 'wrong_side', 'red_light', 'illegal_parking', 'seatbelt', 'triple_riding', 'distracted_driving']
    vehicle_classes = ['motorcycle', 'car', 'truck', 'car', 'motorcycle', 'bus', 'auto_rickshaw', 'car']
    license_plates_prefixes = ['KA-01-AB-', 'KA-03-CD-', 'KA-05-EF-', 'KA-51-GH-', 'KA-53-IJ-', 'KA-02-KL-', 'KA-04-MN-', 'KA-50-OP-']
    statuses = ['pending', 'confirmed', 'dismissed']
    
    now = datetime.now()
    
    print("Seeding traffic violations...")
    for i in range(50):
        v_time = (now - timedelta(hours=random.randint(0, 48), minutes=random.randint(0, 59))).isoformat()
        cam_idx = i % len(camera_ids)
        cam_id = camera_ids[cam_idx]
        v_type = violation_types[i % len(violation_types)]
        v_class = vehicle_classes[i % len(vehicle_classes)]
        plate = license_plates_prefixes[i % len(license_plates_prefixes)] + f"{random.randint(1000, 9999)}"
        conf = 0.75 + random.random() * 0.22
        status = statuses[i % len(statuses)]
        sha256 = "".join(random.choices("0123456789abcdef", k=64))
        
        cursor.execute(
            """INSERT INTO traffic_violations 
               (violation_id, violation_time, camera_id, track_id, vehicle_class, license_plate, plate_confidence, violation_type, confidence, sha256_hash, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), v_time, cam_id, random.randint(100, 999), v_class, plate, 0.8 + random.random() * 0.18, v_type, conf, sha256, status)
        )
    conn.commit()

    # 3. Seed Traffic Density Hourly
    print("Seeding traffic density stats...")
    for cam_id in camera_ids:
        for hour in range(24):
            window_hour = f"{now.strftime('%Y-%m-%d')} {hour:02d}:00:00"
            vehicles = random.randint(10, 150) if hour in range(7, 21) else random.randint(2, 25)
            avg_speed = 35.0 + random.random() * 25.0
            cursor.execute(
                "INSERT INTO traffic_density_hourly (window_hour, camera_id, lane_id, total_vehicles, average_speed) VALUES (?, ?, ?, ?, ?)",
                (window_hour, cam_id, f"Lane-{random.randint(1, 3)}", vehicles, avg_speed)
            )
    conn.commit()

    # 4. Seed Training Runs
    print("Seeding training runs...")
    runs = [
        ("run_idd_lite_yolov12", "YOLOv12-Nano", "idd-lite", 100, "completed", 100, 0.452, 0.084),
        ("run_helmet_yolov12", "YOLOv12-Small", "helmet-dataset", 50, "completed", 50, 0.521, 0.076),
        ("run_plate_yolov12", "YOLOv12-Medium", "plate-dataset", 30, "running", 15, 0.495, 0.092)
    ]
    for r_id, model, dataset, epochs, status, curr_epoch, mAP, loss in runs:
        cursor.execute(
            """INSERT INTO training_runs 
               (run_id, model_name, dataset, epochs, status, current_epoch, best_map, loss, started_at, config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r_id, model, dataset, epochs, status, curr_epoch, mAP, loss, now.isoformat(), '{"batch_size": 16, "img_size": 640}')
        )
    conn.commit()
    conn.close()
    print("Database seeding completed successfully!")

if __name__ == "__main__":
    seed_db()
