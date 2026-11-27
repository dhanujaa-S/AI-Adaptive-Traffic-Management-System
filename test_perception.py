"""Tests for Module 2: YOLOv8 vehicle detection and ROI-based multi-road counting."""

import time

import numpy as np

from src.config import DEFAULT_FRAME_SIZE, MODELS_DIR, ROADS
from src.counter import RoadVehicleCounter
from src.detector import Detection, VehicleDetector, annotate_road_frame
from src.video_manager import MultiRoadVideoManager


def test_roi_counter_filters_outside_detections() -> None:
    """Verify ROI counter counts vehicles inside the zone and ignores outside ones."""
    counter = RoadVehicleCounter()
    frame_shape = (DEFAULT_FRAME_SIZE[1], DEFAULT_FRAME_SIZE[0], 3)  # (360, 640, 3)

    detections = [
        Detection(bbox=(100, 150, 180, 220), class_id=2, class_name="car", confidence=0.9, anchor_point=(140, 220)),
        Detection(bbox=(250, 180, 290, 240), class_id=3, class_name="motorcycle", confidence=0.85, anchor_point=(270, 240)),
        # Outside default ROI (y < 0.20 * 360 = 72)
        Detection(bbox=(200, 10, 250, 50), class_id=5, class_name="bus", confidence=0.92, anchor_point=(225, 50)),
    ]

    result = counter.count_road("North", detections, frame_shape)
    assert result.total_count == 2
    assert result.class_counts["car"] == 1
    assert result.class_counts["motorcycle"] == 1
    assert result.class_counts["bus"] == 0


def test_detector_and_counter_on_sample_videos() -> None:
    """Run real YOLOv8 inference across all 4 sample road frames and verify annotations."""
    model_path = MODELS_DIR / "yolov8n.pt"
    assert model_path.exists(), f"Missing model weights at {model_path}"

    detector = VehicleDetector()
    counter = RoadVehicleCounter()

    with MultiRoadVideoManager(auto_download=False) as vm:
        # Advance several frames to ensure active traffic vehicles are in view
        frames = vm.read_all_frames()
        for _ in range(15):
            frames = vm.read_all_frames()

        t0 = time.perf_counter()
        detections_by_road = detector.detect_batch(frames)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        sample_shape = frames["North"].shape
        counts = counter.count_all_roads(detections_by_road, sample_shape)

        assert set(detections_by_road.keys()) == set(ROADS)
        assert set(counts.keys()) == set(ROADS)

        total_detected_across_roads = sum(c.total_count for c in counts.values())
        assert total_detected_across_roads > 0, "Expected vehicles detected across sample traffic frames"

        for road in ROADS:
            poly = counter.get_roi_polygon(road, sample_shape)
            annotated = annotate_road_frame(
                frames[road],
                counts[road].detections_in_roi,
                roi_polygon=poly,
                road_name=road,
                total_count=counts[road].total_count,
                signal_state="GREEN" if road == "North" else "RED",
            )
            assert annotated.shape == frames[road].shape
            assert isinstance(annotated, np.ndarray)

        print(f"\n[Benchmark] 4-road YOLOv8n batch inference: {elapsed_ms:.1f} ms | Total vehicles in ROI: {total_detected_across_roads}")
