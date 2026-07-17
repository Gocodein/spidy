"""
DETECTOR AI — Endangered Species Detection & Tracking System
=============================================================
A 6-stage AI pipeline for detecting endangered species, tracking
their behavior, and identifying human disturbance in natural habitats.

Target: Bengal Tiger (initial release)
Hardware: Optimized for NVIDIA RTX 4050 (6 GB VRAM)

Stages:
    1. Generic Animal/Human/Vehicle Detector (YOLOv8)
    2. Fine-Grained Species Classifier (EfficientNetV2)
    3. Multi-Object Tracking (ByteTrack)
    4. Behavior Recognition (Trajectory Analysis)
    5. Human-Disturbance Analysis
    6. Logging, Alerting & Dashboard

Author: DETECTOR AI Team
"""

__version__ = "1.0.0"
__project__ = "DETECTOR AI"
__target_species__ = "Bengal Tiger (Panthera tigris tigris)"
