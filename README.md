<div align="center">

# TrafficAI

### Intelligent Transportation System for Automated Traffic Enforcement

An end-to-end, AI-powered multi-violation traffic enforcement platform with real-time video analytics, legal evidence generation, and a premium monitoring dashboard.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://python.org)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org)
[![YOLOv12](https://img.shields.io/badge/YOLOv12-Ultralytics-orange.svg)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)

</div>

---

## Overview

TrafficAI automates traffic violation detection using computer vision and deep learning. It processes live or recorded video from traffic cameras to detect, log, and prosecute violations — generating court-admissible evidence packages under India's Bharatiya Sakshya Adhiniyam (BSA), 2023.

**Key highlights:**

- **8 violation types** detected in real-time
- **7-stage processing pipeline** with deduplication
- **Court-admissible evidence** with SHA-256 hashing & BSA certificates
- **Edge-to-cloud deployment** — runs on laptops to Jetson Orin
- **Premium dark-mode dashboard** with live streaming & analytics

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [ML Training Pipeline](#ml-training-pipeline)
- [Violation Detection](#violation-detection)
- [API Reference](#api-reference)
- [Hardware Compatibility](#hardware-compatibility)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture

```
┌───────────────────────────────┐        ┌─────────────────────────────────────┐
│   React Dashboard (Vite)      │◄──────►│   FastAPI Backend                   │
│                               │  API   │                                     │
│   ┌─────────────────────┐     │        │   ┌─────────────────────────────┐   │
│   │  Dashboard          │     │        │   │  7-Stage Unified Pipeline    │   │
│   │  Violations Table   │     │        │   │  1. Preprocessing            │   │
│   │  Analytics Charts   │     │        │   │  2. YOLO Detection           │   │
│   │  Camera Network     │     │        │   │  3. Object Tracking          │   │
│   │  Training Monitor   │     │        │   │  4. Micro-Classifiers        │   │
│   │  System Settings    │     │        │   │  5. Violation Detection      │   │
│   └─────────────────────┘     │        │   │  6. ANPR (License Plates)    │   │
│                               │        │   │  7. Evidence Generation      │   │
│   SSE ◄── Live Violations     │        │   └─────────────────────────────┘   │
│   MJPEG ◄── Live Camera Feed  │        │                                     │
└───────────────────────────────┘        └────────────┬────────────────────────┘
                                                      │
                                     ┌────────────────┼────────────────┐
                                     │                │                │
                               ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐
                               │  SQLite   │   │  YOLOv12  │   │  Evidence  │
                               │  (async)  │   │  Models   │   │  Storage   │
                               └───────────┘   └───────────┘   └───────────┘
```

### Processing Pipeline

Each video frame flows through a **7-stage cascade**:

| Stage | Component | Purpose |
|-------|-----------|---------|
| 1 | **Frame Preprocessor** | Quality assessment (blur, brightness, noise, haze) + adaptive enhancement (CLAHE, bilateral filter) |
| 2 | **Inference Engine** | YOLOv12 detection with optional SAHI sliced inference for small objects |
| 3 | **Object Tracker** | ByteTrack-inspired multi-object tracker with Kalman-like velocity prediction |
| 4 | **Micro-Classifiers** | ResNet18 for traffic light color, MobileNetV3 for seatbelt detection |
| 5 | **Violation Detector** | Rule engine checking 8 violation types against spatial-temporal conditions |
| 6 | **ANPR Service** | EasyOCR/PaddleOCR with multi-attempt preprocessing and Indian plate format validation |
| 7 | **Evidence Generator** | SHA-256 hashed evidence package + BSA Section 63(4)(c) legal certificate |

---

## Features

### Violation Detection

| Violation | Method | Confidence |
|-----------|--------|------------|
| **Helmet Non-Compliance** | Rider detection + temporal counter | Per-frame YOLO |
| **Triple Riding** | Multi-rider count on motorcycle | Per-frame YOLO |
| **Seatbelt Non-Compliance** | Windshield crop + MobileNetV3 classifier | Classifier score |
| **Speed Violation** | Ground-plane homography + ByteTrack | km/h estimation |
| **Wrong-Side Driving** | Directional flow analysis (cosine similarity) | Direction vector |
| **Red-Light Running** | Centroid crossing stop line during red phase | Signal state |
| **Stop-Line Violation** | Vehicle stopped past stop line | Position + timer |
| **Illegal Parking** | Stationary vehicle in restricted zone | Duration threshold |

### Deduplication (4 Layers)

1. **Temporal sliding window** — 10s per (track_id, violation_type)
2. **License-plate dedup** — 300s per camera+plate+violation_type
3. **Perceptual hash (dHash)** — Hamming distance < 10, 5-min window
4. **Database-level** — Unique constraint on violation records

### Dashboard

- **Real-time violation feed** via Server-Sent Events (SSE)
- **Live camera MJPEG streams** from RTSP cameras
- **Analytics** — 7-day trends, heatmaps, speed histograms, peak hours
- **Camera management** — Add/remove cameras, calibration, zone config
- **Training monitor** — Start/stop training, live loss/mAP curves
- **Settings** — Runtime configuration persisted to database

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- (Optional) NVIDIA GPU with CUDA for faster inference

### 1. Clone & Configure

```bash
git clone https://github.com/harshalsp07/TrafficVAI.git
cd TrafficVAI
cp .env.example .env
```

Edit `.env` to match your setup. The defaults work for CPU-only development.

### 2. Backend

```bash
cd backend
python -m venv venv

# Activate virtual environment
source venv/bin/activate    # Linux / macOS
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at **http://localhost:8000**
API docs at **http://localhost:8000/docs**

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard runs at **http://localhost:5173**

### 4. Seed Demo Data (Optional)

```bash
cd backend
python seed_db.py
```

This populates the database with sample cameras and violations for demonstration.

---

## Docker Deployment

Single command to launch the full stack:

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |

To run in the background:

```bash
docker-compose up -d --build
docker-compose logs -f          # View logs
docker-compose down             # Stop
```

---

## ML Training Pipeline

### Quick Training

```bash
cd training
pip install -r requirements.txt

# Generate a synthetic smoke-test dataset
python scripts/prepare_dataset.py --output datasets/sample --synthetic

# Train vehicle detection
python scripts/train.py --config configs/vehicle_detection.yaml --epochs 10 --device cpu

# Export to ONNX
python scripts/export_model.py --weights runs/best.pt --format onnx

# Evaluate
python scripts/evaluate.py --weights runs/best.pt --data datasets/sample/dataset.yaml
```

### Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | YOLO training with structured JSON progress output |
| `scripts/evaluate.py` | mAP, precision, recall, F1 metrics + FPS benchmarking |
| `scripts/export_model.py` | Export to ONNX, TensorRT, OpenVINO, TorchScript, CoreML |
| `scripts/augment.py` | Albumentations-based augmentation (rain, fog, blur, noise) |
| `scripts/prepare_dataset.py` | Synthetic data generation + COCO-to-YOLO conversion |
| `scripts/merge_datasets.py` | Merge 4 datasets into unified 11-class schema |
| `scripts/convert_voc_to_yolo.py` | Pascal VOC XML → YOLO TXT format converter |
| `scripts/convert_idd_lite.py` | IDD Lite segmentation masks → YOLO bounding boxes |

### Training Configs

| Config | Task | Model | Classes | Epochs |
|--------|------|-------|---------|--------|
| `unified_detection.yaml` | Detect | YOLO11m | 11 | 150 |
| `vehicle_detection.yaml` | Detect | YOLOv12n | 7 | 100 |
| `helmet_detection.yaml` | Detect | YOLOv12s | 4 | 120 |
| `plate_detection.yaml` | Detect | YOLOv12n | 1 | 80 |
| `seatbelt_classifier.yaml` | Classify | YOLO11n-cls | 2 | 60 |
| `traffic_light_classifier.yaml` | Classify | ResNet18 | 3 | 40 |

### Supported Model Variants

| Model | Parameters | mAP@50:95 | CPU Inference | GPU Inference |
|-------|-----------|-----------|---------------|---------------|
| YOLOv12-Nano | 6.5M | ~40.6% | ~50ms | ~3.8ms |
| YOLOv12-Small | 7.5M | ~48.0% | ~80ms | ~5.1ms |
| YOLOv12-Medium | 20.2M | ~52.5% | ~150ms | ~7.2ms |

---

## Violation Detection

### How It Works

1. **Video Input** — RTSP stream, uploaded video file, or webcam
2. **Frame Extraction** — Configurable FPS (default: 1 FPS for detection, 15 FPS for display)
3. **Object Detection** — YOLOv12 detects vehicles, riders, pedestrians, helmets, plates
4. **Multi-Object Tracking** — ByteTrack-inspired tracker assigns persistent IDs across frames
5. **Violation Rules** — Each tracked object is checked against spatial-temporal rules
6. **Evidence Capture** — Violation triggers automatic frame capture + plate OCR
7. **Legal Certificate** — SHA-256 hash + BSA 2023 metadata for court admissibility

### Evidence Package

Each violation generates:

```
evidence/{violation_id}/
├── raw_frame.jpg           # Original captured frame
├── annotated_frame.jpg     # Frame with detection overlays
├── vehicle_crop.jpg        # Cropped vehicle image
├── plate_crop.jpg          # Cropped license plate image
├── metadata.json           # Violation details + timestamps
├── hashes.json             # SHA-256 hashes of all files
└── bsa_certificate.json    # BSA Section 63(4)(c) legal certificate
```

### Legal Compliance (India)

The system generates **BSA Section 63(4)(c)** certificates for electronic evidence admissibility under the **Bharatiya Sakshya Adhiniyam, 2023** (replacing Section 65B of the Indian Evidence Act). SHA-256 hashes are computed on all evidentiary media at capture time to ensure tamper detection.

---

## API Reference

### Violations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/violations` | List violations (paginated, filterable) |
| `GET` | `/api/violations/stats` | Dashboard statistics |
| `GET` | `/api/violations/{id}` | Single violation details |
| `PATCH` | `/api/violations/{id}` | Update violation status |
| `GET` | `/api/violations/{id}/certificate` | BSA legal certificate |

### Cameras

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/cameras` | List all cameras |
| `POST` | `/api/cameras` | Register new camera |
| `PUT` | `/api/cameras/{id}` | Update camera config |
| `DELETE` | `/api/cameras/{id}` | Remove camera |
| `POST` | `/api/cameras/{id}/calibrate` | Homography calibration |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/analytics/trends` | 7-day violation trends |
| `GET` | `/api/analytics/heatmap` | Violation heatmap data |
| `GET` | `/api/analytics/distribution` | By-type distribution |
| `GET` | `/api/analytics/speed` | Speed histogram |
| `GET` | `/api/analytics/density` | Hourly traffic density |
| `GET` | `/api/analytics/peak-hours` | Peak hours heatmap |

### Inference & Streaming

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/inference/video` | Upload video for processing |
| `POST` | `/api/inference/start-stream` | Start RTSP stream processing |
| `GET` | `/api/inference/status` | Active stream status |
| `GET` | `/api/stream/violations` | SSE live violation feed |
| `GET` | `/api/stream/camera/{id}` | MJPEG live camera feed |

### Training & Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/training/start` | Start training run |
| `GET` | `/api/training/runs` | List training history |
| `POST` | `/api/training/deploy` | Deploy trained model |
| `GET` | `/api/reports/summary` | Summary statistics |
| `GET` | `/api/reports/export/csv` | Export violations as CSV |
| `GET` | `/api/reports/export/pdf` | Export report as PDF |

---

## Hardware Compatibility

| Hardware | GPU | Mode | Expected FPS |
|----------|-----|------|-------------|
| Laptop (CPU only) | None | ONNX/OpenVINO | 5–10 |
| Desktop (GTX 1060+) | 6 GB | PyTorch FP16 | 30–60 |
| Workstation (RTX 3060+) | 8 GB+ | TensorRT FP16 | 80–150 |
| Jetson Orin Nano | 8 GB | TensorRT INT8 | 260–312 |
| Jetson AGX Orin | 64 GB | TensorRT INT8 | 400+ |

---

## Project Structure

```
TrafficVAI/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # App entry point (lifespan, CORS, routers)
│   │   ├── config.py          # Pydantic Settings (loaded from .env)
│   │   ├── api/               # REST API route modules
│   │   │   ├── violations.py  # Violation CRUD + stats + certificates
│   │   │   ├── cameras.py     # Camera management + calibration
│   │   │   ├── analytics.py   # Trends, heatmaps, distributions
│   │   │   ├── training.py    # Training job lifecycle
│   │   │   ├── inference.py   # Video upload + stream management
│   │   │   ├── stream.py      # SSE + MJPEG streaming
│   │   │   ├── reports.py     # CSV/PDF export
│   │   │   └── settings.py    # Runtime config persistence
│   │   ├── models/            # Pydantic request/response schemas
│   │   ├── services/          # Core business logic
│   │   │   ├── unified_pipeline.py    # 7-stage cascade
│   │   │   ├── inference_engine.py    # YOLO detection + SAHI
│   │   │   ├── tracker_service.py     # ByteTrack-inspired tracker
│   │   │   ├── violation_detector.py  # 8 violation types
│   │   │   ├── anpr_service.py        # License plate recognition
│   │   │   ├── evidence_generator.py  # Evidence + BSA certificates
│   │   │   └── report_service.py      # PDF/CSV generation
│   │   ├── db/                # Async SQLite (aiosqlite, WAL mode)
│   │   └── utils/             # SHA-256 hashing utilities
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                   # React dashboard (Vite)
│   ├── src/
│   │   ├── pages/             # Dashboard, Violations, Analytics, etc.
│   │   ├── components/        # StatsCard, ViolationBadge, StatusDot
│   │   └── hooks/             # useApi, useSSE
│   ├── package.json
│   └── Dockerfile
├── training/                   # ML training pipeline
│   ├── configs/               # 7 YAML training configs
│   ├── scripts/               # Train, evaluate, export, augment
│   └── datasets/              # Dataset configs (images excluded from git)
├── docker-compose.yml
├── .env.example
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 19, Vite 8, ApexCharts, Leaflet, Framer Motion |
| **Backend** | FastAPI (async), Uvicorn, Pydantic Settings |
| **Database** | SQLite (aiosqlite, WAL mode) |
| **ML Detection** | Ultralytics YOLOv12, SAHI sliced inference |
| **Object Tracking** | ByteTrack-inspired (Kalman prediction, IoU association) |
| **OCR** | EasyOCR / PaddleOCR |
| **Image Processing** | OpenCV, NumPy, Pillow, Albumentations |
| **Micro-Classifiers** | PyTorch (MobileNetV3, ResNet18) |
| **Report Generation** | ReportLab (PDF), matplotlib, CSV |
| **Real-time Streaming** | SSE (sse-starlette), MJPEG |
| **Containerization** | Docker, Docker Compose |

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./weights/yolov12n.pt` | Path to YOLO model weights |
| `DEVICE` | `cpu` | Inference device (`cpu`, `cuda:0`, `openvino`) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence threshold |
| `NMS_THRESHOLD` | `0.45` | Non-max suppression threshold |
| `MAX_STREAMS` | `4` | Maximum concurrent video streams |
| `SPEED_LIMIT_KMH` | `60` | Speed limit for violation detection |
| `HELMET_CONFIDENCE_THRESHOLD` | `0.6` | Helmet detection threshold |
| `PARKING_DURATION_SECONDS` | `300` | Duration before illegal parking alert |
| `ANPR_ENABLED` | `true` | Enable automatic number plate recognition |
| `OCR_ENGINE` | `easyocr` | OCR engine (`easyocr` or `paddleocr`) |
| `EVIDENCE_RETENTION_HOURS` | `72` | Hours to keep evidence files |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed CORS origins |

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for safer roads.**

[Report Bug](https://github.com/harshalsp07/TrafficVAI/issues) · [Request Feature](https://github.com/harshalsp07/TrafficVAI/issues) · [Contributing](CONTRIBUTING.md)

</div>
