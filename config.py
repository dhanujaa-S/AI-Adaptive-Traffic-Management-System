"""Central configuration for the AI-Based Adaptive Traffic Management System."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
MODELS_DIR = BASE_DIR / "models"

ROADS: Tuple[str, ...] = ("North", "South", "East", "West")

# Standard COCO class IDs relevant to Indian urban traffic.
# A custom-trained Indian Traffic (IDD) YOLOv8 model can override this mapping.
COCO_VEHICLE_CLASSES: Dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Indian Roads Congress (IRC-106) inspired Passenger Car Unit (PCU) weights.
DEFAULT_PCU_WEIGHTS: Dict[str, float] = {
    "bicycle": 0.25,
    "motorcycle": 0.5,
    "auto_rickshaw": 1.0,
    "car": 1.0,
    "bus": 3.0,
    "truck": 3.0,
}

# Distinct display colors (BGR for OpenCV) per vehicle class.
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "bicycle": (255, 200, 0),
    "motorcycle": (0, 215, 255),
    "auto_rickshaw": (0, 165, 255),
    "car": (0, 255, 0),
    "bus": (255, 80, 80),
    "truck": (180, 0, 255),
}

DEFAULT_FRAME_SIZE: Tuple[int, int] = (640, 360)  # (width, height)

# Normalized (0.0-1.0) approach ROI polygons for each simulated road.
DEFAULT_ROI_NORMALIZED: Dict[str, List[Tuple[float, float]]] = {
    road: [(0.05, 0.20), (0.95, 0.20), (0.98, 0.96), (0.02, 0.96)]
    for road in ROADS
}

# Publicly available sample traffic videos mapped to the 4 simulated approaches.
# Each entry maps road -> (local_filename, download_url, fallback_start_offset_ratio).
SAMPLE_VIDEO_SOURCES: Dict[str, Dict[str, object]] = {
    "North": {
        "filename": "road_north_traffic.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
        "start_offset_ratio": 0.0,
    },
    "South": {
        "filename": "road_south_traffic.mp4",
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
        "start_offset_ratio": 0.35,
    },
    "East": {
        "filename": "road_east_traffic.mp4",
        "url": "https://raw.githubusercontent.com/sheefanaaz123/Vehicle-Detection-and-Counting/main/Video.mp4",
        "start_offset_ratio": 0.0,
    },
    "West": {
        "filename": "road_west_traffic.mp4",
        "url": "https://raw.githubusercontent.com/sheefanaaz123/Vehicle-Detection-and-Counting/main/Video.mp4",
        "start_offset_ratio": 0.45,
    },
}


@dataclass
class DetectorConfig:
    """Configuration for YOLOv8 vehicle detection."""

    model_path: str = str(MODELS_DIR / "yolov8n.pt")
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    image_size: int = 640
    class_map: Dict[int, str] = field(default_factory=lambda: dict(COCO_VEHICLE_CLASSES))


@dataclass
class DensityConfig:
    """Configuration for PCU-based traffic density estimation."""

    pcu_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PCU_WEIGHTS))
    road_capacity_pcu: Dict[str, float] = field(
        default_factory=lambda: {road: 18.0 for road in ROADS}
    )
    ema_alpha: float = 0.35


@dataclass
class SignalConfig:
    """Configuration for adaptive traffic signal timing."""

    min_green_sec: float = 10.0
    max_green_sec: float = 45.0
    yellow_sec: float = 3.0
    all_red_sec: float = 1.0
    max_wait_sec: float = 90.0
    fixed_green_sec: float = 25.0
    aging_weight: float = 0.015  # Extra priority per second waited
