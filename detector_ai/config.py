"""
DETECTOR AI — Configuration
Central configuration for all pipeline stages.
Optimized for: RTX 4050 (6 GB VRAM), Bengal Tiger detection.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "data"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# MODEL DEFAULTS
# ------------------------------------------------------------------
DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"  # swap to your fine-tuned .pt once trained
DEFAULT_CLASSIFIER_WEIGHTS = None     # set after training species classifier

# ------------------------------------------------------------------
# STAGE 1: DETECTOR — COCO class mappings (used until fine-tuned)
# ------------------------------------------------------------------
COCO_ANIMAL_CLASS_IDS = {15, 16, 17, 18, 19, 20, 21, 22, 23}
# 15=bird, 16=cat, 17=dog, 18=horse, 19=sheep,
# 20=cow, 21=elephant, 22=bear, 23=zebra, 24=giraffe

COCO_HUMAN_CLASS_ID = 0          # 'person'
COCO_VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# After fine-tuning, your model will have its own class map:
TIGER_CLASS_ID = 0
TIGER_CLASS_NAME = "bengal_tiger"

# Detection confidence thresholds
DETECTION_CONF_THRESHOLD = 0.35
SPECIES_CONF_THRESHOLD = 0.50

# ------------------------------------------------------------------
# STAGE 3: TRACKER
# ------------------------------------------------------------------
TRACKER_TYPE = "bytetrack.yaml"   # or "botsort.yaml"
TRACK_PERSIST = True

# ------------------------------------------------------------------
# STAGE 4: BEHAVIOR ESTIMATION
# ------------------------------------------------------------------
BEHAVIOR_HISTORY_WINDOW = 20     # frames of position history per track

# Speed thresholds (pixels/frame — adjust based on resolution & FPS)
SPEED_RESTING = 1.5
SPEED_WALKING = 12.0
SPEED_RUNNING = 25.0

# Direction variance thresholds
DIR_VAR_ALERT = 1.2              # erratic direction → stressed/alert
DIR_VAR_STALKING = 0.3           # very low variance + moderate speed = stalking

# Acceleration threshold
ACCEL_STARTLE = 8.0              # sudden speed increase → startled

# Proximity for group behavior
GROUP_PROXIMITY_PX = 200

# ------------------------------------------------------------------
# STAGE 5: DISTURBANCE ANALYSIS
# ------------------------------------------------------------------
DISTURBANCE_DISTANCE_PX = 250    # pixel distance for proximity alert
DISTURBANCE_CRITICAL_PX = 100    # very close — critical alert

# Behavior shift detection: if animal goes from calm → alert/fleeing
# within this many frames of a human appearing, flag disturbance
BEHAVIOR_SHIFT_WINDOW = 10

# Severity levels
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

# ------------------------------------------------------------------
# STAGE 6: LOGGING
# ------------------------------------------------------------------
DEFAULT_DB_PATH = DATA_DIR / "wildlife_events.db"
LOG_BATCH_SIZE = 30              # commit to DB every N frames (performance)
ENABLE_CONSOLE_ALERTS = True
ENABLE_SOUND_ALERTS = False

# ------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------
# BGR colors for OpenCV
COLOR_ANIMAL_SAFE = (0, 220, 80)       # green
COLOR_ANIMAL_DISTURBED = (0, 50, 255)  # red
COLOR_HUMAN = (255, 140, 0)            # blue-ish
COLOR_VEHICLE = (200, 200, 0)          # cyan-ish
COLOR_ALERT_TEXT = (0, 0, 255)         # red
COLOR_INFO_TEXT = (255, 255, 255)      # white
COLOR_TRACK_LINE = (255, 200, 50)      # trajectory line

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 2
BBOX_THICKNESS = 2

# ------------------------------------------------------------------
# TRAINING — RTX 4050 (6 GB VRAM) optimized
# ------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """YOLOv8 training hyperparameters optimized for RTX 4050 (6GB)."""
    epochs: int = 100
    batch_size: int = 8           # safe for 6GB VRAM at 640px
    img_size: int = 640
    patience: int = 20            # early stopping
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    warmup_epochs: int = 5
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.0          # no vertical flip for animals
    fliplr: float = 0.5
    device: str = "0"            # GPU 0
    amp: bool = True             # mixed precision — critical for 6GB VRAM
    workers: int = 4


@dataclass
class ClassifierConfig:
    """EfficientNetV2-S classifier training config."""
    model_name: str = "tf_efficientnetv2_s"
    img_size: int = 384
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_classes: int = 1         # Bengal Tiger (binary: tiger vs not-tiger)
    device: str = "cuda"
    amp: bool = True


# ------------------------------------------------------------------
# PIPELINE RUNTIME
# ------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Runtime configuration for the detection pipeline."""
    video_source: object = 0                          # webcam by default
    yolo_weights: str = DEFAULT_YOLO_WEIGHTS
    classifier_weights: Optional[str] = DEFAULT_CLASSIFIER_WEIGHTS
    db_path: Path = DEFAULT_DB_PATH
    tracker: str = TRACKER_TYPE
    show_display: bool = True
    save_video: bool = False
    save_video_path: Optional[str] = None
    log_to_db: bool = True
    detection_conf: float = DETECTION_CONF_THRESHOLD
    species_conf: float = SPECIES_CONF_THRESHOLD
    disturbance_distance: int = DISTURBANCE_DISTANCE_PX
    behavior_window: int = BEHAVIOR_HISTORY_WINDOW
    frame_skip: int = 0                               # process every N-th frame (0=all)
