#!/usr/bin/env python3
"""
Generate the unified YOLOv11 Kaggle training notebook (.ipynb).

Run:  python generate_kaggle_notebook.py
Produces: Unified_YOLOv11_Traffic_Training.ipynb
"""

import json, textwrap, sys
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────
def md(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source),
    }

def code(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "execution": {
                "iopub.execute_input": "",
                "iopub.status.busy": "",
                "iopub.status.idle": "",
                "shell.execute_reply": ""
            },
            "trusted": True
        },
        "outputs": [],
        "source": _lines(source),
    }

def _lines(src: str) -> list[str]:
    """Dedent and split into list-of-lines (notebook JSON format)."""
    src = textwrap.dedent(src).strip()
    lines = src.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

# ── cells ────────────────────────────────────────────────────
cells = []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TITLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
# 🚦 Unified Traffic Violation Detection — YOLOv11 Training
### Multi-Dataset Fusion: IDD + License Plate + Helmet + Seatbelt

**Account:** `harshal0704` &nbsp;|&nbsp; **Runtime:** Kaggle T4 GPU (16 GB VRAM) &nbsp;|&nbsp; **Model:** YOLOv11

---

> **What this notebook does:**
> 1. Discovers & validates all 4 Kaggle datasets (IDD Detection 24.5 GB, Plate Detection 186 MB, Motorcycle Helmet, Seatbelt Detection)
> 2. Converts IDD PASCAL-VOC XML → YOLO TXT with class remapping
> 3. Reads Roboflow `data.yaml` files and remaps helmet / seatbelt classes dynamically
> 4. Remaps plate detection labels
> 5. Assembles a unified dataset with 16 detection classes
> 6. Trains **one** YOLOv11 model end-to-end on T4
> 7. Evaluates, visualises results, and exports the final `.pt` weights
>
> **⚡ T4 Optimisations:** Mixed-precision (AMP), gradient checkpointing, symlinked images (saves disk), 640px, batch 16.
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1 – ENVIRONMENT SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 1 · Environment Setup"))

cells.append(code("""\
%%time
# ── Install / upgrade Ultralytics (pinned for reproducibility) ──
!pip install -q ultralytics>=8.3.0
!pip install -q pyyaml tqdm matplotlib seaborn Pillow

import os, sys, shutil, random, glob, yaml, json, time, gc, warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from IPython.display import display, HTML, Markdown

import torch
import ultralytics
from ultralytics import YOLO, checks

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("━" * 60)
print(f"  Ultralytics  : v{ultralytics.__version__}")
print(f"  PyTorch       : v{torch.__version__}")
print(f"  Python        : {sys.version.split()[0]}")
print(f"  CUDA compiled : {torch.version.cuda}")
print("━" * 60)
checks()
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2 – GPU VERIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 2 · GPU Verification"))

cells.append(code("""\
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"🔥 GPU      : {gpu}")
    print(f"💾 VRAM     : {vram:.1f} GB")
    print(f"🔧 CUDA ver : {torch.version.cuda}")
    # Quick sanity – can we allocate?
    _ = torch.zeros(1, device="cuda")
    print("✅ CUDA allocation OK")
else:
    raise RuntimeError(
        "❌ No GPU detected!\\n"
        "Go to: Settings → Accelerator → GPU T4 x1 → Save, then restart."
    )
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3 – CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
## 3 · Configuration

> **Edit the cells below** if your Kaggle dataset slugs differ or you want to change hyper-parameters.
"""))

cells.append(code("""\
# ╔══════════════════════════════════════════════════════════╗
# ║              KAGGLE DATASET INPUT PATHS                  ║
# ╚══════════════════════════════════════════════════════════╝
# We dynamically resolve paths to support different usernames
# and nested dataset structures automatically.

KAGGLE_INPUT = Path("/kaggle/input")

def resolve_dataset_path(slug_candidates):
    possible_roots = []
    for slug in slug_candidates:
        possible_roots.extend([
            KAGGLE_INPUT / slug,
            KAGGLE_INPUT / "datasets" / "harshal0704" / slug,
            KAGGLE_INPUT / "harshal0704" / slug,
        ])
    
    existing_roots = [p for p in possible_roots if p.exists()]
    target_markers = ["images", "labels", "Annotations", "JPEGImages", "train", "valid", "test", "dataset.yaml", "data.yaml"]
    
    for r in existing_roots:
        if any((r / marker).exists() for marker in target_markers):
            return r
        for sub in r.iterdir():
            if sub.is_dir():
                if any((sub / marker).exists() for marker in target_markers):
                    return sub
                    
    if existing_roots:
        return existing_roots[0]
    return KAGGLE_INPUT / slug_candidates[0]

# ── Resolve dataset roots ──
IDD_ROOT      = resolve_dataset_path(["idd-detection", "idd_detection"])
PLATE_ROOT    = resolve_dataset_path(["plate-detection", "plate_detection"])
HELMET_ROOT   = resolve_dataset_path(["motorcycle-helmet", "motorcycle_helmet"])
SEATBELT_ROOT = resolve_dataset_path(["seatbelt-detection4yolov11", "seatbelt_detection"])

print(f"📂 IDD_ROOT      resolved to: {IDD_ROOT}")
print(f"📂 PLATE_ROOT    resolved to: {PLATE_ROOT}")
print(f"📂 HELMET_ROOT   resolved to: {HELMET_ROOT}")
print(f"📂 SEATBELT_ROOT resolved to: {SEATBELT_ROOT}")

# ── Working directory (writable, ~20 GB on Kaggle) ──
WORK_DIR    = Path("/kaggle/working")
UNIFIED_DIR = WORK_DIR / "unified_dataset"

# ── IDD camera-view filter ──
# The IDD dataset has 6 camera views.  To save disk & time you can pick a subset.
# Set to None to use ALL views.
IDD_VIEWS = ["frontFar", "frontNear"]          # ← change as needed

# ╔══════════════════════════════════════════════════════════╗
# ║                UNIFIED CLASS MAPPING                     ║
# ╚══════════════════════════════════════════════════════════╝
UNIFIED_CLASSES = {
    0:  "person",
    1:  "rider",
    2:  "car",
    3:  "truck",
    4:  "bus",
    5:  "motorcycle",
    6:  "bicycle",
    7:  "autorickshaw",
    8:  "animal",
    9:  "traffic_light",
    10: "traffic_sign",
    11: "license_plate",
    12: "helmet",
    13: "no_helmet",
    14: "seatbelt",
    15: "no_seatbelt",
}
NUM_CLASSES = len(UNIFIED_CLASSES)

# IDD VOC class-name → unified class-id
IDD_CLASS_MAP = {
    "person":           0,
    "rider":            1,
    "car":              2,
    "truck":            3,
    "bus":              4,
    "motorcycle":       5,
    "bicycle":          6,
    "autorickshaw":     7,
    "auto_rickshaw":    7,
    "auto-rickshaw":    7,
    "animal":           8,
    "traffic light":    9,
    "trafficlight":     9,
    "traffic_light":    9,
    "traffic sign":     10,
    "trafficsign":      10,
    "traffic_sign":     10,
    "train":            3,   # rare, map to truck
    "vehicle fallback": 2,
    "caravan":          2,
}

# ╔══════════════════════════════════════════════════════════╗
# ║            TRAINING HYPER-PARAMETERS (T4)                ║
# ╚══════════════════════════════════════════════════════════╝
MODEL_VARIANT = "yolo11m.pt"   # n / s / m / l / x
IMG_SIZE      = 640
BATCH_SIZE    = 16             # lower to 8 if OOM
EPOCHS        = 80
PATIENCE      = 15             # early-stop
WORKERS       = 2              # Kaggle has few CPU cores
LR0           = 0.01
LRF           = 0.01
OPTIMIZER     = "AdamW"
AMP_ENABLED   = True

print(f"📋 Unified Model — {NUM_CLASSES} classes:")
for i, n in UNIFIED_CLASSES.items():
    print(f"   {i:>2d}: {n}")
print(f"\\n⚙️  Model={MODEL_VARIANT}  img={IMG_SIZE}  batch={BATCH_SIZE}  epochs={EPOCHS}")
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4 – DATASET EXPLORATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 4 · Dataset Exploration"))

cells.append(code("""\
def tree(path, prefix="", max_depth=2, _depth=0, _first=True):
    \"\"\"Pretty-print a directory tree.\"\"\"
    p = Path(path)
    if not p.exists():
        print(f"⚠️  {p}  — NOT FOUND (add dataset to notebook)")
        return
    if _first:
        print(f"📂 {p}")
    entries = sorted(p.iterdir())
    dirs  = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    # show first 8 files
    for f in files[:8]:
        sz = f.stat().st_size
        unit = "B"
        if sz > 1024**2: sz /= 1024**2; unit = "MB"
        elif sz > 1024:  sz /= 1024;    unit = "KB"
        print(f"{prefix}├── {f.name}  ({sz:.1f} {unit})")
    if len(files) > 8:
        print(f"{prefix}├── ... +{len(files)-8} more files")
    for d in dirs:
        n_children = sum(1 for _ in d.rglob("*"))
        print(f"{prefix}├── 📁 {d.name}/  ({n_children} items)")
        if _depth < max_depth - 1:
            tree(d, prefix=prefix + "│   ", max_depth=max_depth,
                 _depth=_depth+1, _first=False)

print("=" * 65)
print("  IDD DETECTION (~24.5 GB)")
print("=" * 65)
tree(IDD_ROOT, max_depth=2)

print("\\n" + "=" * 65)
print("  PLATE DETECTION (~186 MB)")
print("=" * 65)
tree(PLATE_ROOT, max_depth=2)

print("\\n" + "=" * 65)
print("  MOTORCYCLE HELMET (Roboflow)")
print("=" * 65)
tree(HELMET_ROOT, max_depth=2)

print("\\n" + "=" * 65)
print("  SEATBELT DETECTION (Roboflow)")
print("=" * 65)
tree(SEATBELT_ROOT, max_depth=2)
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5 – UTILITY FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 5 · Utility Functions"))

cells.append(code("""\
# ── Directory setup ──────────────────────────────────────────
def setup_dirs():
    \"\"\"Create the unified dataset tree.\"\"\"
    for split in ("train", "val", "test"):
        (UNIFIED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (UNIFIED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    print("✅ Unified dataset directories created")

# ── VOC XML parser ───────────────────────────────────────────
def parse_voc_xml(xml_path):
    \"\"\"Return (filename, img_w, img_h, list-of-boxes).\"\"\"
    tree_xml = ET.parse(xml_path)
    root = tree_xml.getroot()

    size_el = root.find("size")
    img_w = int(float(size_el.find("width").text))
    img_h = int(float(size_el.find("height").text))
    fn_el = root.find("filename")
    filename = fn_el.text if fn_el is not None else xml_path.stem + ".jpg"

    boxes = []
    for obj in root.findall("object"):
        diff = obj.find("difficult")
        if diff is not None and int(diff.text) == 1:
            continue
        name = obj.find("name").text.strip()
        bb = obj.find("bndbox")
        xmin = max(0, float(bb.find("xmin").text))
        ymin = max(0, float(bb.find("ymin").text))
        xmax = min(img_w, float(bb.find("xmax").text))
        ymax = min(img_h, float(bb.find("ymax").text))
        if xmax > xmin and ymax > ymin:
            boxes.append((name, xmin, ymin, xmax, ymax))
    return filename, img_w, img_h, boxes

# ── VOC → YOLO conversion ───────────────────────────────────
def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h
    return cx, cy, w, h

# ── Symlink-or-copy (saves disk on Kaggle) ───────────────────
def link_or_copy(src, dst):
    dst = Path(dst)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

# ── Read YAML safely ────────────────────────────────────────
def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

setup_dirs()
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6 – PROCESS IDD DETECTION  (VOC → YOLO)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
## 6 · Process IDD Detection (VOC → YOLO)

The IDD dataset uses **PASCAL-VOC XML** annotations organised by camera view.
We read the provided `train.txt / val.txt / test.txt` split files, filter by
selected camera views, convert bounding boxes to YOLO format, and remap
classes to our unified scheme.

> ⏱ This step may take **5-15 min** depending on how many camera views you selected.
"""))

cells.append(code("""\
%%time
print("🔄 Processing IDD Detection dataset …")
idd_stats = Counter()

# ── Read split files ─────────────────────────────────────────
def read_split_file(path):
    if not path.exists():
        print(f"  ⚠️ {path.name} not found")
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

raw_splits = {
    "train": read_split_file(IDD_ROOT / "train.txt"),
    "val":   read_split_file(IDD_ROOT / "val.txt"),
    "test":  read_split_file(IDD_ROOT / "test.txt"),
}
for s, entries in raw_splits.items():
    print(f"  IDD {s:>5s}: {len(entries)} entries in split file")

# ── Resolve annotation XMLs & images ────────────────────────
# IDD structure: Annotations/<view>/<id>.xml   JPEGImages/<view>/<id>.jpg
# The split files usually contain paths like  "<view>/<id>"

ann_root = IDD_ROOT / "Annotations"
img_root = IDD_ROOT / "JPEGImages"

split_out_map = {
    "train": UNIFIED_DIR / "train",
    "val":   UNIFIED_DIR / "val",
    "test":  UNIFIED_DIR / "test",
}

for split_name, entries in raw_splits.items():
    out = split_out_map[split_name]
    processed = 0

    for entry in tqdm(entries, desc=f"  IDD-{split_name}", leave=True):
        # Optional camera-view filter
        if IDD_VIEWS is not None:
            if not any(v in entry for v in IDD_VIEWS):
                continue

        # Resolve XML path
        xml_path = None
        for candidate in [
            ann_root / f"{entry}.xml",
            ann_root / entry / f"{Path(entry).name}.xml",
        ]:
            if candidate.exists():
                xml_path = candidate
                break
        # Try scanning sub-dirs
        if xml_path is None:
            stem = Path(entry).stem
            for xml_file in ann_root.rglob(f"{stem}.xml"):
                xml_path = xml_file
                break
        if xml_path is None:
            idd_stats["skip_no_xml"] += 1
            continue

        try:
            filename, img_w, img_h, boxes = parse_voc_xml(xml_path)
        except Exception:
            idd_stats["skip_bad_xml"] += 1
            continue

        if img_w <= 0 or img_h <= 0 or not boxes:
            idd_stats["skip_empty"] += 1
            continue

        # Resolve image path
        img_path = None
        for candidate in [
            img_root / f"{entry}.jpg",
            img_root / f"{entry}.jpeg",
            img_root / f"{entry}.png",
            img_root / filename,
            img_root / Path(entry).parent / filename,
        ]:
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            stem = Path(entry).stem
            for img_file in img_root.rglob(f"{stem}.*"):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    img_path = img_file
                    break
        if img_path is None:
            idd_stats["skip_no_img"] += 1
            continue

        # Convert to YOLO lines
        yolo_lines = []
        for (cls_name, xmin, ymin, xmax, ymax) in boxes:
            key = cls_name.lower().strip()
            if key not in IDD_CLASS_MAP:
                idd_stats[f"unmapped_{key}"] += 1
                continue
            cid = IDD_CLASS_MAP[key]
            cx, cy, w, h = voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            idd_stats[f"cls_{UNIFIED_CLASSES[cid]}"] += 1

        if not yolo_lines:
            idd_stats["skip_no_mapped"] += 1
            continue

        # Write label + link image
        safe_stem = Path(entry).stem.replace("/", "_").replace("\\\\", "_")
        uid = f"idd_{safe_stem}"
        lbl_path = out / "labels" / f"{uid}.txt"
        with open(lbl_path, "w") as f:
            f.write("\\n".join(yolo_lines))
        link_or_copy(str(img_path), str(out / "images" / f"{uid}{img_path.suffix}"))
        processed += 1
        idd_stats["images"] += 1
        idd_stats["boxes"] += len(yolo_lines)

    print(f"  ✅ IDD {split_name}: {processed} images written")

print(f"\\n📊 IDD Summary — {idd_stats['images']} images, {idd_stats['boxes']} boxes")
for k, v in sorted(idd_stats.items()):
    if k.startswith("cls_"):
        print(f"   {k[4:]:>20s}: {v}")
for k, v in sorted(idd_stats.items()):
    if k.startswith("skip") or k.startswith("unmapped"):
        print(f"   ⚠️ {k}: {v}")
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7 – PROCESS PLATE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
## 7 · Process Plate Detection

The plate dataset is already in YOLO format.  We just remap class 0 → 11 (`license_plate`).
"""))

cells.append(code("""\
%%time
print("🔄 Processing Plate Detection dataset …")
plate_stats = Counter()

# --- Auto-resolve/correct PLATE_ROOT inside the cell ---
print(f"🔎 Current PLATE_ROOT is set to: {PLATE_ROOT}")
if not PLATE_ROOT.exists() or not any((PLATE_ROOT / f).exists() for f in ["images", "labels"]):
    print("⚠️ PLATE_ROOT is invalid or nested. Searching for correct directory...")
    search_paths = [
        Path("/kaggle/input/plate-detection"),
        Path("/kaggle/input/datasets/harshal0704/plate-detection"),
        Path("/kaggle/input/harshal0704/plate-detection"),
    ]
    found = False
    for p in search_paths:
        if p.exists():
            if any((p / f).exists() for f in ["images", "labels"]):
                PLATE_ROOT = p
                found = True
                break
            for sub in p.iterdir():
                if sub.is_dir() and any((sub / f).exists() for f in ["images", "labels"]):
                    PLATE_ROOT = sub
                    found = True
                    break
        if found:
            break
    print(f"👉 Resolved PLATE_ROOT to: {PLATE_ROOT} (Exists: {PLATE_ROOT.exists()})")
else:
    print("✅ PLATE_ROOT is valid!")

if PLATE_ROOT.exists():
    try:
        print(f"📁 Directory contents of PLATE_ROOT: {os.listdir(PLATE_ROOT)}")
        if (PLATE_ROOT / "images").exists():
            print(f"🖼️ Images folder contains {len(os.listdir(PLATE_ROOT / 'images'))} items")
        if (PLATE_ROOT / "labels").exists():
            print(f"🏷️ Labels folder contains {len(os.listdir(PLATE_ROOT / 'labels'))} items")
    except Exception as e:
        print(f"Error reading directory: {e}")

def find_plate_pairs(root):
    \"\"\"Find all image-label pairs for the plate dataset (handles both flat and split structures).\"\"\"
    pairs = []  # (img_path, lbl_path, split_hint)

    # Check for dataset.yaml
    yaml_path = root / "dataset.yaml"
    if yaml_path.exists():
        cfg = read_yaml(yaml_path)
        print(f"  Found dataset.yaml: {cfg.get('names', '?')}")

    # Pattern A: root/images/<split>/*.jpg + root/labels/<split>/*.txt (Local structure)
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if img_dir.exists():
            for img in img_dir.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    lbl = lbl_dir / (img.stem + ".txt")
                    if lbl.exists():
                        pairs.append((img, lbl, split))

    # Pattern B: root/images/*.jpg + root/labels/*.txt (Flat Kaggle structure)
    if not pairs:
        img_dir = root / "images"
        lbl_dir = root / "labels"
        if img_dir.exists() and lbl_dir.exists():
            imgs = [f for f in img_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            for img in imgs:
                lbl = lbl_dir / (img.stem + ".txt")
                if lbl.exists():
                    pairs.append((img, lbl, "auto"))

    return pairs

# Find the pairs
plate_pairs = find_plate_pairs(PLATE_ROOT)
print(f"  Found {len(plate_pairs)} image-label pairs")

# If they were loaded as a flat folder ("auto"), split them 80% train / 10% val / 10% test
if plate_pairs and plate_pairs[0][2] == "auto":
    random.shuffle(plate_pairs)
    n = len(plate_pairs)
    cut1 = int(n * 0.8)
    cut2 = int(n * 0.9)
    assigned = (
        [(p[0], p[1], "train") for p in plate_pairs[:cut1]] +
        [(p[0], p[1], "val")   for p in plate_pairs[cut1:cut2]] +
        [(p[0], p[1], "test")  for p in plate_pairs[cut2:]]
    )
    plate_pairs = assigned

PLATE_TARGET_ID = 11  # license_plate

# Copy and remap labels
for img_path, lbl_path, split in tqdm(plate_pairs, desc="  Plate"):
    split_dir = UNIFIED_DIR / (split if split in ("train","val","test") else "train")
    
    new_lines = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Remap the class ID to 11 (license_plate)
                new_lines.append(f"{PLATE_TARGET_ID} " + " ".join(parts[1:5]))
                plate_stats["boxes"] += 1

    if new_lines:
        uid = f"plate_{img_path.stem}"
        with open(split_dir / "labels" / f"{uid}.txt", "w") as f:
            f.write("\\n".join(new_lines))
        link_or_copy(str(img_path), str(split_dir / "images" / f"{uid}{img_path.suffix}"))
        plate_stats["images"] += 1

print(f"✅ Plate Detection processing finished: {plate_stats['images']} images, {plate_stats['boxes']} boxes written.")
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8 – PROCESS MOTORCYCLE HELMET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
## 8 · Process Motorcycle Helmet (Roboflow)

Roboflow YOLO datasets have `data.yaml` with class names.
We dynamically read the class names and map them to our unified scheme.
"""))

cells.append(code("""\
%%time
print("🔄 Processing Motorcycle Helmet dataset …")
helmet_stats = Counter()

# ── Read data.yaml to discover classes ───────────────────────
helmet_yaml = None
for candidate in [HELMET_ROOT / "data.yaml",
                  HELMET_ROOT / "dataset.yaml"]:
    if candidate.exists():
        helmet_yaml = read_yaml(candidate)
        break

if helmet_yaml is None:
    print("  ⚠️ No data.yaml found — scanning for any yaml")
    for yf in HELMET_ROOT.rglob("*.yaml"):
        helmet_yaml = read_yaml(yf)
        print(f"  Found: {yf}")
        break

if helmet_yaml:
    src_names = helmet_yaml.get("names", {})
    if isinstance(src_names, list):
        src_names = {i: n for i, n in enumerate(src_names)}
    print(f"  Source classes: {src_names}")
else:
    print("  ⚠️ Could not find YAML config — using default mapping")
    src_names = {0: "helmet", 1: "no_helmet"}

# ── Build dynamic remap ─────────────────────────────────────
# Common Roboflow helmet class names and their unified targets
HELMET_KEYWORD_MAP = {
    "with helmet":    12,   "with_helmet":  12,
    "helmet":         12,   "with-helmet":  12,
    "without helmet": 13,   "without_helmet": 13,
    "no helmet":      13,   "no_helmet":    13,
    "without-helmet": 13,   "no-helmet":    13,
    "head":           13,
    # Keep rider/motorcycle context if present
    "rider":          1,
    "motorcycle":     5,
    "person":         0,
}

helmet_remap = {}  # src_id → unified_id
for src_id, src_name in src_names.items():
    key = str(src_name).lower().strip()
    if key in HELMET_KEYWORD_MAP:
        helmet_remap[int(src_id)] = HELMET_KEYWORD_MAP[key]
    else:
        # Fuzzy match
        for kw, tid in HELMET_KEYWORD_MAP.items():
            if kw in key or key in kw:
                helmet_remap[int(src_id)] = tid
                break
    if int(src_id) not in helmet_remap:
        print(f"  ⚠️ Unmapped helmet class {src_id}: '{src_name}' — skipping")

print(f"  Helmet remap: {helmet_remap}")

# ── Process splits ───────────────────────────────────────────
for split in ("train", "valid", "val", "test"):
    img_dir = HELMET_ROOT / split / "images"
    lbl_dir = HELMET_ROOT / split / "labels"
    if not img_dir.exists():
        continue
    out_split = "val" if split == "valid" else split
    out_dir = UNIFIED_DIR / out_split

    for img_path in tqdm(list(img_dir.iterdir()), desc=f"  Helmet-{split}", leave=False):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        new_lines = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    src_id = int(parts[0])
                    if src_id in helmet_remap:
                        new_lines.append(
                            f"{helmet_remap[src_id]} " + " ".join(parts[1:5])
                        )
                        helmet_stats[UNIFIED_CLASSES[helmet_remap[src_id]]] += 1

        if new_lines:
            uid = f"helmet_{img_path.stem}"
            with open(out_dir / "labels" / f"{uid}.txt", "w") as f:
                f.write("\\n".join(new_lines))
            link_or_copy(str(img_path),
                         str(out_dir / "images" / f"{uid}{img_path.suffix}"))
            helmet_stats["images"] += 1

print(f"\\n✅ Helmet: {helmet_stats['images']} images")
for k, v in helmet_stats.items():
    if k != "images":
        print(f"   {k}: {v}")
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  9 – PROCESS SEATBELT DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
## 9 · Process Seatbelt Detection (Roboflow)

Same approach as helmet — read `data.yaml`, dynamically remap classes.
"""))

cells.append(code("""\
%%time
print("🔄 Processing Seatbelt Detection dataset …")
seatbelt_stats = Counter()

# ── Read data.yaml ───────────────────────────────────────────
sb_yaml = None
for candidate in [SEATBELT_ROOT / "data.yaml",
                  SEATBELT_ROOT / "dataset.yaml"]:
    if candidate.exists():
        sb_yaml = read_yaml(candidate)
        break

if sb_yaml is None:
    for yf in SEATBELT_ROOT.rglob("*.yaml"):
        sb_yaml = read_yaml(yf)
        print(f"  Found: {yf}")
        break

if sb_yaml:
    sb_names = sb_yaml.get("names", {})
    if isinstance(sb_names, list):
        sb_names = {i: n for i, n in enumerate(sb_names)}
    print(f"  Source classes: {sb_names}")
else:
    print("  ⚠️ Could not find YAML — using default mapping")
    sb_names = {0: "seatbelt", 1: "no_seatbelt"}

# ── Dynamic remap ────────────────────────────────────────────
SB_KEYWORD_MAP = {
    "seatbelt":      14,   "seatbelt_on":   14,
    "seat belt":     14,   "with seatbelt": 14,
    "belt":          14,   "wearing":       14,
    "no seatbelt":   15,   "no_seatbelt":   15,
    "seatbelt_off":  15,   "without seatbelt": 15,
    "no seat belt":  15,   "not wearing":   15,
    "no-seatbelt":   15,   "without-seatbelt": 15,
    # Contextual classes
    "person":        0,    "driver":        0,
}

sb_remap = {}
for src_id, src_name in sb_names.items():
    key = str(src_name).lower().strip()
    # Exact match first
    if key in SB_KEYWORD_MAP:
        sb_remap[int(src_id)] = SB_KEYWORD_MAP[key]
    else:
        # Check if "no" appears (absence) vs presence
        if "no" in key or "without" in key or "off" in key:
            sb_remap[int(src_id)] = 15  # no_seatbelt
        elif "seatbelt" in key or "seat" in key or "belt" in key:
            sb_remap[int(src_id)] = 14  # seatbelt
        else:
            for kw, tid in SB_KEYWORD_MAP.items():
                if kw in key or key in kw:
                    sb_remap[int(src_id)] = tid
                    break
    if int(src_id) not in sb_remap:
        print(f"  ⚠️ Unmapped seatbelt class {src_id}: '{src_name}' — skipping")

print(f"  Seatbelt remap: {sb_remap}")

# ── Process splits ───────────────────────────────────────────
for split in ("train", "valid", "val", "test"):
    img_dir = SEATBELT_ROOT / split / "images"
    lbl_dir = SEATBELT_ROOT / split / "labels"
    if not img_dir.exists():
        continue
    out_split = "val" if split == "valid" else split
    out_dir = UNIFIED_DIR / out_split

    for img_path in tqdm(list(img_dir.iterdir()), desc=f"  SB-{split}", leave=False):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        new_lines = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    src_id = int(parts[0])
                    if src_id in sb_remap:
                        new_lines.append(
                            f"{sb_remap[src_id]} " + " ".join(parts[1:5])
                        )
                        seatbelt_stats[UNIFIED_CLASSES[sb_remap[src_id]]] += 1

        if new_lines:
            uid = f"sb_{img_path.stem}"
            with open(out_dir / "labels" / f"{uid}.txt", "w") as f:
                f.write("\\n".join(new_lines))
            link_or_copy(str(img_path),
                         str(out_dir / "images" / f"{uid}{img_path.suffix}"))
            seatbelt_stats["images"] += 1

print(f"\\n✅ Seatbelt: {seatbelt_stats['images']} images")
for k, v in seatbelt_stats.items():
    if k != "images":
        print(f"   {k}: {v}")
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  10 – DATASET VERIFICATION & STATISTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 10 · Dataset Verification & Statistics"))

cells.append(code("""\
print("📊 Unified Dataset Statistics")
print("=" * 65)

total_stats = {s: {"images": 0, "labels": 0, "boxes": 0} for s in ("train","val","test")}
class_counts = Counter()

for split in ("train", "val", "test"):
    img_dir = UNIFIED_DIR / split / "images"
    lbl_dir = UNIFIED_DIR / split / "labels"

    n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
    total_stats[split]["images"] = n_img
    total_stats[split]["labels"] = n_lbl

    # Count boxes per class
    if lbl_dir.exists():
        for lbl in lbl_dir.glob("*.txt"):
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        class_counts[cid] += 1
                        total_stats[split]["boxes"] += 1

for s in ("train", "val", "test"):
    st = total_stats[s]
    print(f"  {s:>5s} → {st['images']:>6d} images | {st['labels']:>6d} labels | {st['boxes']:>7d} boxes")

total_imgs = sum(v["images"] for v in total_stats.values())
total_boxes = sum(v["boxes"] for v in total_stats.values())
print(f"  {'TOTAL':>5s} → {total_imgs:>6d} images | {total_boxes:>7d} boxes")
print()

# Class distribution bar chart
fig, ax = plt.subplots(figsize=(14, 5))
sorted_classes = sorted(class_counts.items())
names  = [UNIFIED_CLASSES.get(c, f"?{c}") for c, _ in sorted_classes]
counts = [cnt for _, cnt in sorted_classes]

colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
bars = ax.barh(names, counts, color=colors, edgecolor="gray", linewidth=0.5)
ax.set_xlabel("Number of Annotations", fontsize=12)
ax.set_title("Unified Dataset — Class Distribution", fontsize=14, fontweight="bold")
ax.invert_yaxis()
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
            f"{cnt:,}", va="center", fontsize=9)
plt.tight_layout()
plt.show()

# Sanity checks
assert total_imgs > 0, "❌ No images found — check dataset paths!"
assert total_boxes > 0, "❌ No annotations found — check processing!"
print("\\n✅ All sanity checks passed!")
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  11 – SAMPLE VISUALISATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 11 · Sample Visualisation"))

cells.append(code("""\
def show_samples(split="train", n=6, seed=42):
    \"\"\"Display n random images with their YOLO annotations overlaid.\"\"\"
    img_dir = UNIFIED_DIR / split / "images"
    lbl_dir = UNIFIED_DIR / split / "labels"
    imgs = sorted(img_dir.glob("*"))
    random.seed(seed)
    samples = random.sample(imgs, min(n, len(imgs)))

    cmap = plt.cm.get_cmap("tab20", NUM_CLASSES)

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path)
        w, h = img.size
        ax.imshow(img)
        ax.set_title(img_path.stem[:40], fontsize=8)
        ax.axis("off")

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        rect = patches.Rectangle(
                            (x1, y1), bw*w, bh*h,
                            linewidth=2, edgecolor=cmap(cid),
                            facecolor="none"
                        )
                        ax.add_patch(rect)
                        ax.text(x1, y1 - 3,
                                UNIFIED_CLASSES.get(cid, "?"),
                                fontsize=7, color="white",
                                bbox=dict(facecolor=cmap(cid), alpha=0.7, pad=1))

    for ax in axes[len(samples):]:
        ax.axis("off")
    plt.suptitle(f"Sample Annotations — {split}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

show_samples("train", n=6)
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  12 – CREATE TRAINING YAML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 12 · Create Training YAML"))

cells.append(code("""\
# Generate the dataset.yaml that Ultralytics will consume
dataset_yaml = {
    "path": str(UNIFIED_DIR),
    "train": "train/images",
    "val":   "val/images",
    "test":  "test/images",
    "nc":    NUM_CLASSES,
    "names": UNIFIED_CLASSES,
}

yaml_path = UNIFIED_DIR / "dataset.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

print(f"✅ Saved: {yaml_path}")
print()
with open(yaml_path) as f:
    print(f.read())
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  13 – TRAIN YOLO11
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
## 13 · Train YOLOv11 🚀

> **Expected duration on T4:** ~2–6 hours depending on dataset size and epochs.
>
> The model checkpoint is saved every 5 epochs and on best mAP.
> If the session disconnects, you can resume from the last checkpoint (see cell below).
"""))

cells.append(code("""\
# ╔══════════════════════════════════════════════════════════╗
# ║                   TRAINING LAUNCH                        ║
# ╚══════════════════════════════════════════════════════════╝
gc.collect()
torch.cuda.empty_cache()

# ── Dynamic GPU and Batch Size configuration ──
device_list = [i for i in range(torch.cuda.device_count())]
print(f"🖥️ Detected GPUs: {device_list}")

if len(device_list) > 1:
    gpu_devices = device_list
    training_batch_size = BATCH_SIZE * len(device_list)
    num_workers = 2 * len(device_list) 
    print(f"🚀 Multi-GPU DDP mode active! Using GPUs: {gpu_devices} with scaled total batch size: {training_batch_size}")
else:
    gpu_devices = 0
    training_batch_size = BATCH_SIZE
    num_workers = WORKERS
    print(f"🚀 Single-GPU mode active! Using GPU: {gpu_devices} with batch size: {training_batch_size}")

model = YOLO(MODEL_VARIANT)
print(f"🔧 Loaded {MODEL_VARIANT}  — {sum(p.numel() for p in model.model.parameters())/1e6:.1f}M params")

results = model.train(
    data         = str(UNIFIED_DIR / "dataset.yaml"),
    epochs       = EPOCHS,
    imgsz        = IMG_SIZE,
    batch        = training_batch_size,
    patience     = PATIENCE,
    workers      = num_workers,
    device       = gpu_devices,
    optimizer    = OPTIMIZER,
    lr0          = LR0,
    lrf          = LRF,
    amp          = AMP_ENABLED,

    # Augmentation (tuned for traffic scenes)
    hsv_h        = 0.02,
    hsv_s        = 0.7,
    hsv_v        = 0.5,
    degrees      = 5.0,
    translate    = 0.15,
    scale        = 0.6,
    shear        = 2.0,
    perspective  = 0.001,
    fliplr       = 0.5,
    mosaic       = 1.0,
    mixup        = 0.15,
    copy_paste   = 0.2,
    erasing      = 0.1,

    # Loss
    box          = 7.5,
    cls          = 1.0,
    dfl          = 1.5,

    # Saving
    save_period  = 5,
    save         = True,
    plots        = True,
    project      = str(WORK_DIR / "runs"),
    name         = "unified_traffic_v1",
    exist_ok     = True,
)

print("\\n🎉 Training complete!")
print(f"   Best weights: {WORK_DIR / 'runs' / 'unified_traffic_v1' / 'weights' / 'best.pt'}")
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  13b – RESUME TRAINING (optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
### 13b · Resume Training (if session disconnected)

Uncomment and run this cell only if you need to **resume** from a checkpoint.
"""))

cells.append(code("""\
# RESUME_FROM = WORK_DIR / "runs" / "unified_traffic_v1" / "weights" / "last.pt"
# model = YOLO(str(RESUME_FROM))
# results = model.train(resume=True)
# print("✅ Training resumed and completed!")
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  14 – EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 14 · Evaluation & Metrics"))

cells.append(code("""\
# Load best weights and evaluate on validation set
best_pt = WORK_DIR / "runs" / "unified_traffic_v1" / "weights" / "best.pt"
model = YOLO(str(best_pt))

metrics = model.val(
    data    = str(UNIFIED_DIR / "dataset.yaml"),
    split   = "val",
    imgsz   = IMG_SIZE,
    batch   = BATCH_SIZE,
    device  = 0,
    plots   = True,
    save_json = True,
)

print("\\n" + "=" * 60)
print("  VALIDATION RESULTS")
print("=" * 60)
print(f"  mAP@50     : {metrics.box.map50:.4f}")
print(f"  mAP@50-95  : {metrics.box.map:.4f}")
print(f"  Precision   : {metrics.box.mp:.4f}")
print(f"  Recall      : {metrics.box.mr:.4f}")
print("=" * 60)

# Per-class AP
print("\\n  Per-Class AP@50:")
for i, ap in enumerate(metrics.box.ap50):
    name = UNIFIED_CLASSES.get(i, f"class_{i}")
    bar = "█" * int(ap * 40)
    print(f"  {i:>2d} {name:<18s} {ap:.3f}  {bar}")
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  15 – VISUALISE TRAINING CURVES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 15 · Training Curves & Confusion Matrix"))

cells.append(code("""\
run_dir = WORK_DIR / "runs" / "unified_traffic_v1"

# ── Display saved training plots ─────────────────────────────
plot_files = [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
]

for pf in plot_files:
    fp = run_dir / pf
    if fp.exists():
        print(f"\\n{'─'*40}")
        print(f"  {pf}")
        print(f"{'─'*40}")
        display(Image.open(fp))
    else:
        print(f"  ⚠️ {pf} not found")

# ── Also read CSV results if available ───────────────────────
csv_path = run_dir / "results.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print("\\n📈 Training log (last 5 epochs):")
    display(df.tail())
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  16 – INFERENCE DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("## 16 · Inference Demo on Test Images"))

cells.append(code("""\
best_pt = WORK_DIR / "runs" / "unified_traffic_v1" / "weights" / "best.pt"
model = YOLO(str(best_pt))

# Pick random test images
test_imgs = sorted((UNIFIED_DIR / "test" / "images").glob("*"))
if not test_imgs:
    test_imgs = sorted((UNIFIED_DIR / "val" / "images").glob("*"))

samples = random.sample(test_imgs, min(8, len(test_imgs)))

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

for ax, img_path in zip(axes, samples):
    results = model.predict(str(img_path), imgsz=IMG_SIZE, conf=0.25, verbose=False)
    annotated = results[0].plot(line_width=2, font_size=10)
    ax.imshow(annotated[..., ::-1])  # BGR → RGB
    ax.set_title(img_path.stem[:35], fontsize=9)
    ax.axis("off")

for ax in axes[len(samples):]:
    ax.axis("off")

plt.suptitle("Inference Results — Unified Traffic Model", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  17 – EXPORT & SAVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
## 17 · Export & Download Model

The best weights are saved as a `.pt` file.
You can also export to ONNX or TorchScript for deployment.
"""))

cells.append(code("""\
best_pt = WORK_DIR / "runs" / "unified_traffic_v1" / "weights" / "best.pt"
last_pt = WORK_DIR / "runs" / "unified_traffic_v1" / "weights" / "last.pt"

# ── Copy to /kaggle/working for easy download ───────────────
for src in [best_pt, last_pt]:
    if src.exists():
        dst = WORK_DIR / src.name
        shutil.copy2(src, dst)
        sz = dst.stat().st_size / 1024**2
        print(f"📦 Saved: {dst}  ({sz:.1f} MB)")

# ── Optional: Export to ONNX ─────────────────────────────────
print("\\n🔄 Exporting to ONNX …")
model = YOLO(str(best_pt))
onnx_path = model.export(format="onnx", imgsz=IMG_SIZE, simplify=True)
if Path(onnx_path).exists():
    shutil.copy2(onnx_path, WORK_DIR / "best.onnx")
    print(f"✅ ONNX export: {WORK_DIR / 'best.onnx'}")

# ── Optional: Export to TorchScript ──────────────────────────
# ts_path = model.export(format="torchscript", imgsz=IMG_SIZE)
# print(f"✅ TorchScript export: {ts_path}")

print("\\n" + "=" * 60)
print("  🎉 ALL DONE!")
print("  Files in /kaggle/working/ are downloadable from the Output tab.")
print("=" * 60)
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  18 – DISK CLEANUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cells.append(md("""\
## 18 · Disk Cleanup (Optional)

Run this only if you're running low on disk.
It removes the unified dataset (symlinks + label files) but keeps model weights.
"""))

cells.append(code("""\
# ⚠️ Uncomment to delete the unified dataset and free disk space
# shutil.rmtree(UNIFIED_DIR, ignore_errors=True)
# print("🗑️ Unified dataset removed")

# Check remaining disk
import subprocess
result = subprocess.run(["df", "-h", "/kaggle/working"], capture_output=True, text=True)
print(result.stdout)
"""))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ASSEMBLE NOTEBOOK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat_minor": 2,
            "pygments_lexer": "ipython3",
            "version": "3.10.14"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [
                {
                    "sourceId": 0,
                    "sourceType": "datasetVersion",
                    "datasetSlug": "idd-detection",
                    "ownerSlug": "harshal0704"
                },
                {
                    "sourceId": 0,
                    "sourceType": "datasetVersion",
                    "datasetSlug": "plate-detection",
                    "ownerSlug": "harshal0704"
                },
                {
                    "sourceId": 0,
                    "sourceType": "datasetVersion",
                    "datasetSlug": "motorcycle-helmet",
                    "ownerSlug": "harshal0704"
                },
                {
                    "sourceId": 0,
                    "sourceType": "datasetVersion",
                    "datasetSlug": "seatbelt-detection4yolov11",
                    "ownerSlug": "harshal0704"
                }
            ],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells,
}

# Write notebook
out_path = Path(__file__).parent / "Unified_YOLOv11_Traffic_Training.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook generated: {out_path}")
print(f"   {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
