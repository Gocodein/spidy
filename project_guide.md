# 🐯 DETECTOR AI — Complete Project Guide

Welcome to the **DETECTOR AI** project! This document serves as your complete guide, detailing the architecture, essential knowledge, setup instructions, and the implementation plan for the upcoming phases.

---

## 🧠 Basic Details You Should Know

Before proceeding with the training phase, here are the key technical details about your system:

> [!TIP]
> **Hardware Optimization**
> The entire pipeline and training scripts have been explicitly optimized for your **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)**. We are using a `batch_size` of `8` and Automatic Mixed Precision (`AMP=True`) to ensure you don't run out of memory during training and inference.

1. **Python Environment**: We are using Python 3.10 in an isolated virtual environment located at `.venv310/`. All dependencies (PyTorch with CUDA 12.1, Ultralytics, Timm, Streamlit, etc.) are already installed here.
2. **6-Stage Architecture**: Your AI pipeline processes every video frame through 6 distinct stages:
   * **Stage 1 (Detector)**: YOLOv8 finds animals, humans, and vehicles.
   * **Stage 2 (Classifier)**: EfficientNetV2-S identifies the specific species (Bengal Tiger).
   * **Stage 3 (Tracker)**: ByteTrack assigns persistent IDs to follow objects across frames.
   * **Stage 4 (Behavior)**: Analyzes movement to classify states (Resting, Walking, Running, Alert, Fleeing, Stalking).
   * **Stage 5 (Disturbance)**: Flags human-wildlife proximity and sudden behavior shifts.
   * **Stage 6 (Logging)**: Saves all data to a local SQLite database (`data/wildlife_events.db`).
3. **Data Sourcing**: The Bengal Tiger dataset is not manually downloaded; instead, we have a script (`setup_tiger_dataset.py`) that fetches raw camera trap data from **GBIF** (Rajaji National Park) and **LILA BC** (WCS Camera Traps), automatically generating bounding boxes using MegaDetector.

---

## 🛠️ Setup & Usage Instructions

Since the code and environment are already built, here is how you interact with the system locally.

### 1. Activating the Environment
Whenever you open a new terminal to work on this project, you must activate the virtual environment:
```powershell
# On Windows PowerShell
.\.venv310\Scripts\activate
```

### 2. Running the Live Pipeline (Inference)
You can test the pipeline right now using the pre-trained general YOLO model (it will detect general animals/humans until we train the tiger model).
```powershell
# Run with webcam
python run_detector.py --source 0 --show

# Run on a saved video and save the output
python run_detector.py --source path/to/video.mp4 --save-video output.mp4
```

### 3. Viewing the Research Dashboard
The dashboard visualizes the data logged by Stage 6.
```powershell
streamlit run dashboard/app.py
```

---

## 🚀 Implementation Plan (Next Steps)

The core code is complete and has passed all smoke tests. The next major phase is **Data Preparation & Model Training**. 

> [!IMPORTANT]
> The upcoming dataset download will fetch a large amount of data (potentially several Gigabytes). Ensure you have a stable internet connection and sufficient disk space before starting Step 1.

### Step 1: Download & Prepare the Dataset
We will run the automated dataset setup script. This fetches images from wildlife databases, runs a base AI to draw boxes around animals, and formats it for YOLOv8.
* **Command**: `python training/setup_tiger_dataset.py`
* **Output**: A `tiger_dataset/` folder containing `images/`, `labels/`, and `tiger_data.yaml`.

### Step 2: Quality Assurance (QA) Review
Before training, we must ensure the auto-generated bounding boxes actually wrap around tigers correctly.
* **Command**: `python training/qa_review_boxes.py`
* **Action**: This generates an HTML file. You will open it in your browser, scroll through the images, and delete any bad images from the dataset folders.

### Step 3: Train the YOLOv8 Detector (Stage 1)
Fine-tune the YOLOv8 model to specifically recognize tigers and humans in wildlife settings.
* **Command**: `python training/train_detector.py --data tiger_dataset/tiger_data.yaml`
* **Output**: The best model weights will be saved to `models/tiger_best.pt`.

### Step 4: Train the Species Classifier (Stage 2)
Train the EfficientNetV2-S model on cropped images of tigers vs. non-tigers to improve classification accuracy.
* **Command**: `python training/train_classifier.py --data training/tiger_dataset/crops`
* **Output**: The best model weights will be saved to `models/species_classifier.pth`.

### Step 5: Final Integration Test
Run the main pipeline using your newly trained, specialized models.
* **Command**: `python run_detector.py -m models/tiger_best.pt --species-model models/species_classifier.pth --show`
