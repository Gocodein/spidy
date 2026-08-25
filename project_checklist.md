# ✅ DETECTOR AI — Status & Action Checklist

Use this checklist to track our progress. Everything in **Phase 1 & 2** is fully completed and tested. We are now moving into **Phase 3: Data & Training**.

## Phase 1: Foundation & Architecture (DONE)
- `[x]` Set up Python 3.10 virtual environment (`.venv310`).
- `[x]` Install PyTorch with CUDA 12.1 support for RTX 4050.
- `[x]` Install all ML dependencies (Ultralytics, Timm, OpenCV, Streamlit, etc.).
- `[x]` Create global configuration (`detector_ai/config.py`) optimized for 6GB VRAM.
- `[x]` Create `.gitignore` to prevent pushing large datasets/models to GitHub.

## Phase 2: Core Application Development (DONE)
- `[x]` **Stage 1**: YOLOv8 generic detector with ByteTrack integration (`stage1_detector.py`).
- `[x]` **Stage 2**: EfficientNetV2-S fine-grained species classifier (`stage2_classifier.py`).
- `[x]` **Stage 3**: Multi-object tracker maintaining trajectory history (`stage3_tracker.py`).
- `[x]` **Stage 4**: Behavior kinematics estimator (Resting, Alert, Fleeing, etc.) (`stage4_behavior.py`).
- `[x]` **Stage 5**: Disturbance analyzer based on proximity and behavior shifts (`stage5_disturbance.py`).
- `[x]` **Stage 6**: SQLite database logger & colored console Alert Manager (`stage6_logging.py`).
- `[x]` **Pipeline**: Main orchestrator linking all stages and drawing the HUD (`pipeline.py`).
- `[x]` **CLI**: Command-line interface `run_detector.py` to easily launch the system.
- `[x]` **Dashboard**: Streamlit app for researchers to review database logs (`dashboard/app.py`).
- `[x]` **Smoke Tests**: Verified all modules load and process data successfully using GPU acceleration.

---

## Phase 3: Data & Training (PROCEED WITH THIS)

> **Note:** You will run these commands on your local machine using the activated `.venv310` environment.

- `[ ]` **1. Download & Format Dataset**
  - Run: `python training/setup_tiger_dataset.py`
  - *Wait for GBIF and LILA BC downloads to complete.*
- `[ ]` **2. QA Review Bounding Boxes**
  - Run: `python training/qa_review_boxes.py`
  - *Open the generated HTML file and inspect the bounding boxes.*
- `[ ]` **3. Train YOLOv8 Detector**
  - Run: `python training/train_detector.py --data tiger_dataset/tiger_data.yaml`
  - *Wait for 50 epochs to complete (or early stopping).*
- `[ ]` **4. Train Species Classifier**
  - Run: `python training/train_classifier.py --data training/tiger_dataset/crops`
  - *Wait for training to complete.*

## Phase 4: Final Deployment (PENDING)
- `[ ]` **1. Run Full Pipeline with Custom Models**
  - Run: `python run_detector.py -m models/tiger_best.pt --species-model models/species_classifier.pth --show`
- `[ ]` **2. Verify Disturbance Alerts**
  - *Test with a video containing both humans and tigers to trigger a disturbance alert.*
- `[ ]` **3. Push to GitHub**
  - *Commit and push the final codebase (datasets and models will be ignored automatically).*
