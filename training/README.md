# TrafficAI Training & Architecture Guide

This guide details the machine learning architecture, training algorithms, dataset conversion, and the engineering approach used for the **TrafficAI** Intelligent Transportation System (ITS).

---

## 🏗️ System Architecture & Training Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Preparation & Pipeline                  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Image Preprocessing &                        │
│                Albumentations Augmentation                      │
│      (Rain, Fog, Motion Blur, Rotation, Brightness/Contrast)    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Object Detection Core                       │
│        (YOLOv12 Nano/Small/Medium - Transfer Learning)          │
└────────────────┬───────────────┼────────────────┬───────────────┘
                 │               │                │
                 ▼               ▼                ▼
         [Vehicle Detector]  [Helmet Detector]  [Plate Detector]
           (7 Classes)         (4 Classes)        (1 Class)
                 │               │                │
                 └───────────────┼────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Violation Detection Engine                    │
│      (Homography Calibration, Tracking, Line-Crossing, VLM)     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ANPR OCR Engine                           │
│                 (EasyOCR / PaddleOCR text)                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Evidence Generation                        │
│       (SHA-256 Hashing, BSA 2023 Section 63(4)(c) Certificate)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎨 1. Image Preprocessing & Augmentation

To handle real-world challenges such as rain, fog, nighttime conditions, lens glare, and motion blur, the training pipeline utilizes an automated augmentation engine ([augment.py](scripts/augment.py)) built on **Albumentations**.

*   **Environmental Simulation:**
    *   `RandomRain` and `RandomFog` to make the detector resilient during storms and winters.
    *   `RandomBrightnessContrast` and `HueSaturationValue` to account for changing sunlight, shadows, and low-light environments.
*   **Camera Artifacts:**
    *   `GaussianBlur` and `MotionBlur` to handle fast-moving vehicle motion blur.
    *   `GaussNoise` to simulate sensor noise in low-light CCTV/IP camera feeds.
*   **Geometric Transformations:**
    *   `HorizontalFlip`, `RandomScale`, and `ShiftScaleRotate` to ensure scale-invariance and simulate different camera mounting heights/angles (e.g. traffic light pole vs. gantry).

---

## 🚗 2. Vehicle & Road User Detection

A unified multi-object detection model is trained to recognize and track vehicles and pedestrians:
*   **Model Backbone:** YOLOv12 (supports `yolov12n`, `yolov12s`, and `yolov12m`).
*   **Classes (7):** `car`, `truck`, `motorcycle`, `bus`, `auto_rickshaw`, `bicycle`, `pedestrian`.
*   **Approach:** Transfer learning using pretrained COCO weights. Fine-tuning is customized using [vehicle_detection.yaml](configs/vehicle_detection.yaml) which configures anchor-free bounding box loss, distribution focal loss (`dfl`), and box regression.

---

## 🚨 3. Traffic Violation Detection & Classification

Violations are detected in real-time through the cooperation of the YOLO detection outputs, a tracking algorithm, and geometric calibration:

*   **Helmet Non-Compliance:**
    *   *Algorithm:* The vehicle detector locates motorcycles. For each motorcycle, the area is cropped and analyzed by a secondary **Helmet Detector** (classes: `helmet`, `no_helmet`, `rider`, `motorcycle`).
    *   *Rule:* If a `rider` is detected on a `motorcycle` without a `helmet`, a violation is logged.
*   **Speed Violations:**
    *   *Algorithm:* Pixel coordinates are mapped to 3D world coordinates using a **Homography Calibration Matrix** (configured in the camera settings).
    *   *Rule:* Speeds are calculated across sequential frames using a ByteTrack-based tracker. If the estimated speed exceeds `SPEED_LIMIT_KMH`, a violation is generated.
*   **Wrong-Side Driving:**
    *   *Algorithm:* Vector angles of vehicle tracks are compared against the configured lane direction vector.
    *   *Rule:* If the cosine similarity of the vehicle's direction and the lane direction is less than `WRONG_SIDE_COSINE_THRESHOLD` (e.g., opposite direction), a violation is flag-triggered.
*   **Stop-Line & Red-Light Violations:**
    *   *Algorithm:* A virtual polyline representing the intersection stop-line is defined.
    *   *Rule:* If the traffic light status (received via API or VLM) is **RED** and a vehicle's bounding box crosses the stopline, a violation is registered.
*   **Illegal Parking:**
    *   *Algorithm:* Tracks parked/stopped vehicles in designated "no-parking" polygons.
    *   *Rule:* If a vehicle remains stationary within the polygon for longer than `PARKING_DURATION_SECONDS`, it triggers an alert.
*   **Triple Riding:**
    *   *Algorithm:* Counts the number of `rider` or `pedestrian` bounding boxes overlapping a single `motorcycle` bounding box.
    *   *Rule:* If count $\ge 3$, a triple riding violation is created.
*   **Seatbelt Non-Compliance & Distracted Driving:**
    *   *Algorithm:* Analyzed using a secondary cabin classifier on close-up windshield crops. It is recommended to use dedicated Near-Infrared (NIR) front-facing cameras for this task due to windshield glare on standard intersection cameras.

---

## 🔢 4. License Plate Recognition (ANPR)

The ANPR pipeline is split into two steps:
1.  **Plate Detection:** A fine-tuned YOLOv12 model ([plate_detection.yaml](configs/plate_detection.yaml)) detects the location of the license plate within the vehicle's bounding box.
2.  **OCR Character Extraction:** The cropped plate image is passed to a lightweight OCR engine (**EasyOCR** or **PaddleOCR**). The extracted text is then validated against standard registration patterns (e.g., Indian state formats like `^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}$`) using regular expressions.

---

## 🔒 5. Evidence & Legal Compliance (BSA 2023)

To ensure that the generated alerts are admissible in a court of law (complying with Indian Bharatiya Sakshya Adhiniyam, 2023, Section 63(4)(c) for electronic records):
1.  **Immediate Hashing:** As soon as a violation is detected, a SHA-256 hash is generated for both the raw frame and the cropped violation/plate image.
2.  **Metadata Anchoring:** The hash, timestamp, camera ID, and violation parameters are written to the database.
3.  **Certificate Generation:** The system outputs an encrypted, signed digital certificate containing the SHA-256 hash, ensuring the media has not been tampered with or modified.

---

## 📊 6. Analytics & Performance Evaluation

*   **Analytics Engine:** Aggregates database records to output daily/hourly violation trends, distribution by type, and camera-specific hot-spots.
*   **Evaluation Metrics:**
    *   **mAP@50-95:** The primary metric for object detection, ensuring tight bounding boxes.
    *   **Precision & Recall:** To minimize false positives (e.g. mistaking a backpack for no-helmet) while maintaining high detection rates.
    *   **F1-Score:** The harmonic mean of precision and recall.
    *   **Inference Latency (ms):** Measured during evaluation to ensure the model meets the real-time processing requirements (FPS target).

---

## 🛠️ How to Prepare and Convert Your Custom Datasets

If you have downloaded the datasets provided, you can use the [convert_voc_to_yolo.py](scripts/convert_voc_to_yolo.py) script to prepare them for training:

### A. Convert the License Plate Dataset
This dataset has XML annotations containing the plate characters. The script will map all labels to a single `license_plate` class:
```bash
python training/scripts/convert_voc_to_yolo.py \
  --xml-dir training/datasets/raw_number_plate \
  --out-dir training/datasets/plate_detection \
  --mode plate \
  --split 0.8
```

### B. Convert the Safety Helmet Dataset
This dataset contains `helmet` and `head` XML annotations. The script will map `helmet` to class `0` and `head` (unprotected) to class `1` (`no_helmet`):
```bash
python training/scripts/convert_voc_to_yolo.py \
  --xml-dir training/datasets/raw_safety_helmet \
  --out-dir training/datasets/helmet_detection \
  --mode helmet \
  --split 0.8
```

### C. Convert the IDD Lite Dataset
This dataset contains semantic segmentation masks. The converter extracts connected components for classes `2` (living_thing) and `3` (vehicle) and translates them into YOLO bounding box format:
```bash
python training/scripts/convert_idd_lite.py
```

---

## 🚀 How to Run Training (Colab or Local GPU)

1.  **Configure your dataset path** in the respective config file (e.g., in `training/configs/plate_detection.yaml` set `data: datasets/plate_detection/dataset.yaml`).
2.  **Run the training script:**
    *   **To Train License Plates Detector:**
        ```bash
        python training/scripts/train.py \
          --config training/configs/plate_detection.yaml \
          --epochs 100 \
          --batch 16 \
          --device 0 \
          --model yolov12n.pt
        ```
    *   **To Train Safety Helmet Detector:**
        ```bash
        python training/scripts/train.py \
          --config training/configs/helmet_detection.yaml \
          --epochs 120 \
          --batch 16 \
          --device 0 \
          --model yolov12s.pt
        ```
    *   **To Train IDD Lite Detector:**
        ```bash
        python training/scripts/train.py \
          --config training/configs/idd_lite_detection.yaml \
          --epochs 80 \
          --batch 16 \
          --device 0 \
          --model yolov12n.pt
        ```
3.  **Evaluate:**
    ```bash
    python training/scripts/evaluate.py \
      --weights training/runs/<run_name>/weights/best.pt \
      --data training/datasets/plate_detection/dataset.yaml \
      --benchmark \
      --device 0
    ```
4.  **Export:**
    ```bash
    python training/scripts/export_model.py \
      --weights training/runs/<run_name>/weights/best.pt \
      --format onnx \
      --validate
    ```
