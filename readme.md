# 🐯 DETECTOR AI

> **Real-time endangered species detection, tracking, and human-disturbance analysis using deep learning.**

DETECTOR AI is a 6-stage AI pipeline designed for wildlife conservation researchers. It detects endangered animals (initially Bengal Tigers) in video feeds, tracks their behavior, identifies human disturbance, and logs everything to a searchable database with a live research dashboard.

Optimized for **NVIDIA RTX 4050 (6 GB VRAM)** with mixed-precision inference.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DETECTOR AI PIPELINE                        │
├────────┬────────┬────────┬────────┬─────────────┬──────────────────┤
│        │        │        │        │             │                  │
│  Stage 1  Stage 2  Stage 3  Stage 4    Stage 5      Stage 6       │
│        │        │        │        │             │                  │
│ ┌──────┴─┐ ┌───┴────┐ ┌┴──────┐ ┌┴──────────┐ ┌┴───────────┐ ┌──┴────────┐
│ │ YOLOv8 │ │EffNet  │ │ Byte  │ │ Behavior  │ │ Disturbance│ │ SQLite DB │
│ │Detector│→│V2-S    │→│Track  │→│ Estimator │→│  Analyzer  │→│ + Alerts  │
│ │        │ │Classif.│ │       │ │           │ │            │ │ + Dashbd  │
│ └────────┘ └────────┘ └───────┘ └───────────┘ └────────────┘ └───────────┘
│                                                                     │
│ Frame ──→ Detections ──→ Tracks ──→ Behaviors ──→ Events ──→ Logs  │
└─────────────────────────────────────────────────────────────────────┘
```

| Stage | Module | Model / Method | Output |
|-------|--------|---------------|--------|
| 1 | `stage1_detector.py` | YOLOv8n (fine-tuned) | Bounding boxes + class |
| 2 | `stage2_classifier.py` | EfficientNetV2-S (timm) | Species label + confidence |
| 3 | `stage3_tracker.py` | ByteTrack / BoTSORT | Persistent track IDs |
| 4 | `stage4_behavior.py` | Trajectory kinematics | Behavior state (resting/walking/running/alert/fleeing) |
| 5 | `stage5_disturbance.py` | Proximity + behavior shift | Disturbance events with severity |
| 6 | `stage6_logging.py` | SQLite + AlertManager | Persistent logs + real-time alerts |

---

## Features

- 🎯 **Real-time Detection** — YOLOv8-based animal/human/vehicle detection at 30+ FPS
- 🐯 **Species Classification** — Fine-grained EfficientNetV2-S classifier for endangered species
- 🔗 **Multi-Object Tracking** — ByteTrack/BoTSORT for persistent identity across frames
- 🧠 **Behavior Recognition** — Trajectory-based behavior classification (resting, walking, running, alert, fleeing, stalking)
- ⚠️ **Disturbance Detection** — Automatic human-wildlife proximity and behavior-shift analysis
- 📊 **Research Dashboard** — Interactive Streamlit + Plotly dashboard for data exploration
- 💾 **Persistent Logging** — SQLite database with full event history
- 🖥️ **Live HUD** — Annotated video overlay with FPS, track counts, trajectory lines, and disturbance banners
- ⚡ **RTX 4050 Optimized** — Mixed-precision (AMP) throughout for 6 GB VRAM

---

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/your-org/detector-ai.git
cd detector-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install PyTorch with CUDA (RTX 4050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Detection

```bash
# Webcam (default)
python run_detector.py

# Video file
python run_detector.py -s wildlife_clip.mp4

# With fine-tuned model
python run_detector.py -m models/tiger_detector_best.pt --species-model models/species_classifier_best.pth

# RTSP camera stream
python run_detector.py -s rtsp://192.168.1.100:8554/live

# Save annotated output
python run_detector.py -s input.mp4 --save-video output.mp4

# All options
python run_detector.py --help
```

### 3. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) to explore detection data, species analytics, behavior patterns, and disturbance events.

---

## Training

### Dataset Setup

```bash
# Download and prepare the Bengal Tiger dataset
# (requires internet access to GBIF, LILA BC, and MegaDetector)
python training/setup_tiger_dataset.py

# Review auto-generated bounding boxes
python training/qa_review_boxes.py flag    # generates HTML review sheet
# ... review in browser, export rejected.txt ...
python training/qa_review_boxes.py apply   # removes bad boxes
```

### Fine-Tune Detector (YOLOv8)

```bash
python training/train_detector.py --data tiger_dataset/tiger_data.yaml

# Custom settings
python training/train_detector.py \
    --data tiger_dataset/tiger_data.yaml \
    --model yolov8s.pt \
    --epochs 150 \
    --batch 8 \
    --imgsz 640
```

### Fine-Tune Species Classifier (EfficientNetV2-S)

```bash
# Prepare crops directory:
#   training/tiger_dataset/crops/bengal_tiger/  (positive)
#   training/tiger_dataset/crops/not_tiger/     (negative)

python training/train_classifier.py --data training/tiger_dataset/crops

# Custom settings
python training/train_classifier.py \
    --data training/tiger_dataset/crops \
    --epochs 30 \
    --batch 16 \
    --lr 1e-4
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 (6 GB) | **NVIDIA RTX 4050 (6 GB)** |
| CUDA | 11.8+ | 12.1+ |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB (models + DB) | 20 GB (with datasets) |
| CPU | 4 cores | 8+ cores |
| Python | 3.10+ | 3.10 |

> **Note:** Mixed-precision (AMP) is enabled by default and is critical for fitting within 6 GB VRAM during both training and inference.

---

## Project Structure

```
Detector/
├── detector_ai/                  # Core pipeline package
│   ├── __init__.py
│   ├── config.py                 # Central configuration & dataclasses
│   ├── pipeline.py               # Main pipeline orchestrator
│   ├── stage1_detector.py        # YOLOv8 animal/human/vehicle detector
│   ├── stage2_classifier.py      # EfficientNetV2-S species classifier
│   ├── stage3_tracker.py         # ByteTrack multi-object tracker
│   ├── stage4_behavior.py        # Trajectory-based behavior estimator
│   ├── stage5_disturbance.py     # Human-disturbance analyser
│   └── stage6_logging.py         # SQLite logging & alert manager
│
├── training/                     # Training scripts & data preparation
│   ├── setup_tiger_dataset.py    # Dataset download & YOLO formatting
│   ├── qa_review_boxes.py        # Bounding box QA review tool
│   ├── train_detector.py         # YOLOv8 fine-tuning script
│   └── train_classifier.py       # EfficientNetV2-S training script
│
├── dashboard/                    # Research dashboard
│   └── app.py                    # Streamlit + Plotly dashboard
│
├── models/                       # Trained model weights (gitignored)
├── data/                         # SQLite database & runtime data
├── runs/                         # Training run outputs (gitignored)
│
├── run_detector.py               # CLI entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## CLI Reference

```
usage: run_detector [-h] [-s SOURCE] [-m MODEL] [--species-model PATH]
                    [--db DB] [--show | --no-show] [--save-video PATH]
                    [--conf FLOAT] [--disturbance-dist INT]
                    [--frame-skip N] [-v]

DETECTOR AI — Real-time endangered species detection & tracking.

options:
  -s, --source        Video source: 0 (webcam), file path, or RTSP URL
  -m, --model         YOLO weights path (default: yolov8n.pt)
  --species-model     Species classifier weights (default: None)
  --db                SQLite database path (default: data/wildlife_events.db)
  --show / --no-show  Display live window (default: --show)
  --save-video        Save annotated output video
  --conf              Detection confidence threshold (default: 0.35)
  --disturbance-dist  Disturbance distance in pixels (default: 250)
  --frame-skip        Process every N-th frame (default: 0 = all)
  -v, --verbose       Enable debug logging
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Credits & Data Sources

| Resource | Usage |
|----------|-------|
| [LILA BC](https://lila.science) | Wildlife camera trap datasets (WCS Camera Traps) |
| [GBIF](https://www.gbif.org) | Rajaji National Park tiger occurrence data |
| [Microsoft MegaDetector](https://github.com/microsoft/CameraTraps) | Auto-labeling bounding boxes for weakly-labeled images |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection backbone |
| [timm](https://github.com/huggingface/pytorch-image-models) | EfficientNetV2-S pre-trained weights |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | Multi-object tracking algorithm |

---

<p align="center">
  <b>🐯 Built for wildlife conservation researchers 🌿</b>
</p>
