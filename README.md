---
title: Spidy Wildlife Detector
emoji: 🐾
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: 1.39.0
app_file: dashboard/app.py
pinned: false
license: apache-2.0
---

# 🐾 DETECTOR AI — Multi-Species Wildlife Detection & Monitoring System

> **Real-time 9-class endangered species detection, multi-object tracking, behavior kinematics, and human-disturbance analysis using deep learning.**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA%2012.1-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-8.4.127-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Patent Protected](https://img.shields.io/badge/Patent-Indian%20Patent%20Protected-green.svg)](#-license--patent-notice)

**DETECTOR AI** is an end-to-end 6-stage AI pipeline engineered for wildlife conservation researchers, national park authorities, and automated camera traps. It detects 8 major endangered animal species (plus humans), tracks their movements persistently, classifies fine-grained behavior states, analyzes potential human-wildlife disturbance events in real time, and logs structured analytics to an interactive dashboard.

Optimized for **NVIDIA GeForce RTX 4050 (6 GB VRAM)** with automatic mixed-precision (AMP).

---

## 🎯 Target Species & Classes

The system detects and differentiates across **9 distinct classes**, including rare color morphs:

| Class ID | Species / Entity | Scientific Name | Special Morphs Handled |
| :---: | :--- | :--- | :--- |
| `0` | **Bengal Tiger** | *Panthera tigris tigris* | Includes White Tiger morph |
| `1` | **Asian Elephant** | *Elephas maximus* | Adult & juvenile herd profiles |
| `2` | **Leopard** | *Panthera pardus* | Includes Melanistic (Black Panther) morph |
| `3` | **Greater One-Horned Rhinoceros** | *Rhinoceros unicornis* | Typical camera-trap and habitat poses |
| `4` | **Person / Human** | *Homo sapiens* | Disturbance triggers & perimeter alerts |
| `5` | **Cheetah** | *Acinonyx jubatus* | Distinct spot pattern & slender build |
| `6` | **Jaguar** | *Panthera onca* | Rosette pattern recognition |
| `7` | **Snow Leopard** | *Panthera uncia* | Mountain camouflage & thick coat features |
| `8` | **Sloth Bear** | *Melursus ursinus* | Characteristic chest mark & shaggy fur |

---

## 📊 Benchmark & Model Performance

### Stage 1: YOLOv8n Multi-Species Detector (`models/multispecies_best.pt`)
- **Parameters**: 3.01M (6.2 MB)
- **Validation Dataset**: 1,053 images / 1,441 bounding box instances

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **All Classes (Overall)** | **86.7%** | **76.6%** | **83.4%** | **68.6%** |
| Sloth Bear | 96.5% | 88.7% | 94.1% | 83.5% |
| Jaguar | 94.6% | 92.7% | 95.2% | 79.0% |
| Snow Leopard | 84.4% | 82.9% | 85.0% | 78.4% |
| Asian Elephant | 88.9% | 76.5% | 86.3% | 73.2% |
| Rhinoceros | 90.6% | 76.8% | 84.1% | 72.5% |
| Cheetah | 85.8% | 78.8% | 82.6% | 68.0% |
| Bengal Tiger | 79.7% | 69.4% | 78.4% | 60.7% |
| Leopard | 81.2% | 65.4% | 73.7% | 59.8% |
| Person | 78.8% | 58.3% | 71.4% | 42.5% |

### Stage 2: EfficientNetV2-S Species Classifier (`models/species_classifier_best.pth`)
- **Architecture**: `tf_efficientnetv2_s` (20.19M parameters, 78 MB)
- **Dataset**: 9,517 bounding-box crops across 9 classes
- **Top-1 Validation Accuracy**: **`97.32%`**
- **Inference Speed**: ~4.5 ms per crop (GPU)

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     DETECTOR AI PIPELINE                                       │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┬─────────────────┤
│   Stage 1    │   Stage 2    │   Stage 3    │   Stage 4    │     Stage 5      │     Stage 6     │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────────┼─────────────────┤
│ ┌──────────┐ │ ┌──────────┐ │ ┌──────────┐ │ ┌──────────┐ │ ┌──────────────┐ │ ┌─────────────┐ │
│ │  YOLOv8  │ │ │Efficient-│ │ │  Byte-   │ │ │ Behavior │ │ │ Disturbance  │ │ │ SQLite DB   │ │
│ │ Detector │→│ │  NetV2-S │→│ │  Track   │→│ │Kinematics│→│ │   Analyzer   │→│ │ + Alerts    │ │
│ └──────────┘ │ └──────────┘ │ └──────────┘ │ └──────────┘ │ └──────────────┘ │ │ + Dashboard │ │
│              │              │              │              │                  │ └─────────────┘ │
│ Frame ───────┴─→ Crop ──────┴─→ Trajectory─┴─→ States ────┴─→ Proximity Event─┴─→ SQLite Logs  │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Module | Implementation | Function |
|:---:|---|---|---|
| **1** | `stage1_detector.py` | YOLOv8n (9 classes) | Real-time object localization and initial bounding box detection |
| **2** | `stage2_classifier.py` | EfficientNetV2-S | Fine-grained species classification and verification on high-res crops |
| **3** | `stage3_tracker.py` | ByteTrack / BoTSORT | Multi-object tracking with persistent identity across frames |
| **4** | `stage4_behavior.py` | Trajectory Kinematics | Real-time state estimation: *Resting*, *Walking*, *Running/Fleeing*, *Alert/Pacing*, *Stalking*, *Observing* |
| **5** | `stage5_disturbance.py` | Spatial Proximity & Dynamics | Evaluates species-specific proximity buffers against humans and flags sudden behavioral shifts |
| **6** | `stage6_logging.py` | SQLite3 + Console Alerts | Structured logging of tracks, kinematics, and disturbance events with instant notifications |

---

## ✨ Features

- ⚡ **High FPS Pipeline**: 30–45+ FPS real-time processing on consumer RTX GPUs.
- 🎯 **2-Stage Cascaded Precision**: YOLOv8 locates entities; EfficientNetV2-S verifies subtle markings (e.g., jaguar rosettes vs leopard spots).
- 🧭 **Behavior Kinematics**: Computes velocity, directional variance, acceleration, and dwell time across track histories.
- ⚠️ **Species-Specific Disturbance Radii**: Configurable threshold buffers (e.g., 350px for elephants, 250px for solitary big cats).
- 🖥️ **Live HUD Overlay**: Real-time bounding boxes (green=safe, red=disturbed, blue=human), trajectory tails, and alert banners.
- 📊 **Research Dashboard**: Interactive Streamlit + Plotly interface for spatial heatmaps, transition matrices, and CSV data export.

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/Gocodein/spidy.git
cd spidy

# Create and activate virtual environment
python -m venv .venv310
.venv310\Scripts\activate        # Windows
# source .venv310/bin/activate   # Linux / macOS

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Live Detection & Video Processing

```bash
# 1. Run live webcam feed (loads 9-class detector and classifier by default)
python run_detector.py --source 0 --show

# 2. Process a recorded wildlife video file
python run_detector.py --source wildlife_sample.mp4 --show

# 3. Save annotated output with HUD overlay
python run_detector.py --source input.mp4 --save-video output_annotated.mp4

# 4. Connect to an RTSP IP camera stream
python run_detector.py --source rtsp://192.168.1.100:8554/live --show

# View all CLI options:
python run_detector.py --help
```

### 3. Launch Research Dashboard

```bash
streamlit run dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to review session stats, behavior timelines, disturbance frequency, and export event logs.

---

## 🔬 Dataset & Training Pipeline

The project includes automated pipelines for dataset acquisition, auto-labeling, and model fine-tuning:

```bash
# 1. Download multi-species training data from iNaturalist, WCS, and COCO
python training/download_phase5_data.py

# 2. Fill class distribution gaps via Wikimedia Commons and relaxed filters
python training/fill_dataset_gaps.py

# 3. Train YOLOv8n detector (120 epochs)
python training/train_detector.py --data multispecies_dataset/data.yaml --epochs 120 --name phase5_9class

# 4. Extract bounding-box crops for classifier
python training/extract_crops.py

# 5. Train EfficientNetV2-S species classifier (50 epochs)
python training/train_classifier.py --data multispecies_dataset/crops --epochs 50
```

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **GPU** | NVIDIA GTX 1660 (6 GB) | **NVIDIA RTX 4050 / RTX 3060 (6 GB+)** |
| **CUDA** | 11.8+ | **12.1+** |
| **System RAM** | 8 GB | **16 GB** |
| **Storage** | 2 GB (code + weights) | **10 GB** (with full training datasets) |
| **Python** | 3.10+ | **3.10.11** |

---

## 📁 Repository Structure

```
Detector/
├── detector_ai/                     # Core pipeline package
│   ├── __init__.py
│   ├── config.py                    # 9-class configuration & spatial thresholds
│   ├── pipeline.py                  # 6-stage pipeline orchestrator & HUD renderer
│   ├── stage1_detector.py           # YOLOv8 9-class object detector
│   ├── stage2_classifier.py         # EfficientNetV2-S fine-grained species classifier
│   ├── stage3_tracker.py            # ByteTrack multi-object persistent tracking
│   ├── stage4_behavior.py           # Kinematics behavior estimator
│   ├── stage5_disturbance.py        # Species-specific disturbance analyzer
│   └── stage6_logging.py            # SQLite database logger & console alerts
│
├── training/                        # Training & dataset generation scripts
│   ├── download_phase5_data.py      # Primary dataset downloader (iNat, WCS, COCO)
│   ├── fill_dataset_gaps.py         # Gap filler & auto-annotator (Wikimedia, iNat)
│   ├── extract_crops.py             # Bounding box crop extractor for Stage 2
│   ├── train_detector.py            # YOLOv8 detector fine-tuning script
│   └── train_classifier.py          # EfficientNetV2-S classifier training script
│
├── dashboard/                       # Research dashboard
│   └── app.py                       # Streamlit interactive analysis tool
│
├── models/                          # Trained model checkpoints (gitignored)
│   ├── multispecies_best.pt         # Best 9-class YOLOv8 weights (mAP50=83.4%)
│   └── species_classifier_best.pth  # Best 9-class EfficientNetV2-S weights (acc=97.32%)
│
├── run_detector.py                  # Main CLI entry point
├── requirements.txt                 # Python dependencies
├── project_guide.md                 # Technical architecture guide
├── project_checklist.md             # Project milestones and status
└── README.md                        # Documentation
```

---

## 📜 License & Patent Notice

This project is licensed under the **Apache License, Version 2.0**. See the [`LICENSE`](LICENSE) file for the full license text.

> **🏛️ Patent & Intellectual Property Notice:**  
> The biomimetic ground-level robotic monitoring platform (*Arachnid Research Companion / Spidy*), multi-stage AI detection pipeline, kinematics-based behavior estimation, and human-disturbance monitoring architecture are **officially published and protected under the Indian Patent Office**:
>
> - **Invention Title**: *“Arachnid Research Companion (ARC): A Biomimetic Hexapod Robot for Ground-Level Environmental Monitoring”*
> - **Application Number**: `202531071175 A`
> - **Filing Date**: `26/07/2025` | **Publication Date**: `01/08/2025`
> - **Patent Journal**: *The Patent Office Journal No. 31/2025 (Page 74978)*
> - **Applicant**: JIS College of Engineering
> - **Inventors**: Sagar Shaw, Rajat Mitra, Roshan Kumar Yadav, Sahin Molla
>
> All rights not expressly granted under the Apache 2.0 license are reserved. See [`NOTICE`](NOTICE) for details.

---

## 🤝 Credits & Acknowledgments

- **[iNaturalist](https://www.inaturalist.org)** — High-resolution biodiversity photographic observations.
- **[LILA BC & WCS](https://lila.science/datasets/wcscameratraps)** — Camera trap datasets for wildlife conservation.
- **[COCO Dataset](https://cocodataset.org)** — Diverse human localization annotations.
- **[Wikimedia Commons](https://commons.wikimedia.org)** — Open educational wildlife media repositories.
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — State-of-the-art real-time detection framework.
- **[Ross Wightman / PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)** — EfficientNetV2 backbones.

---

<p align="center">
  <b>🐾 Developed for Automated Wildlife Conservation & Research 🌿</b>
</p>
