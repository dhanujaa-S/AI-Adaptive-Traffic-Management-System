"""End-to-end simulation engine integrating video streams, YOLOv8, PCU density, and signal control."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.config import ROADS, DensityConfig, DetectorConfig, SignalConfig
from src.counter import RoadCountResult, RoadVehicleCounter
from src.density import RoadDensityReport, TrafficDensityEstimator
from src.detector import Detection, VehicleDetector, annotate_road_frame
from src.signal_controller import AdaptiveSignalController, SignalSnapshot
from src.video_manager import MultiRoadVideoManager


@dataclass
class SimulationStepOutput:
    """Complete output of a single simulation step across all 4 roads."""

    step_index: int
    elapsed_sim_sec: float
    annotated_frames_rgb: Dict[str, np.ndarray]
    counts_by_road: Dict[str, RoadCountResult]
    density_reports: Dict[str, RoadDensityReport]
    signal_snapshot: SignalSnapshot
    inference_ms: float


class TrafficSimulationEngine:
    """Orchestrates the 4-road Indian adaptive traffic management simulation."""

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        detector_config: Optional[DetectorConfig] = None,
        density_config: Optional[DensityConfig] = None,
        signal_config: Optional[SignalConfig] = None,
        auto_download_videos: bool = True,
        detect_every_n: int = 1,
    ) -> None:
        self.detector = detector or VehicleDetector(detector_config or DetectorConfig())
        self.counter = RoadVehicleCounter()
        self.density_estimator = TrafficDensityEstimator(density_config or DensityConfig())
        self.signal_controller = AdaptiveSignalController(signal_config or SignalConfig())
        self.video_manager = MultiRoadVideoManager(auto_download=auto_download_videos)

        self.detect_every_n = max(1, detect_every_n)
        self.step_index = 0
        self.elapsed_sim_sec = 0.0
        self._last_detections: Dict[str, List[Detection]] = {road: [] for road in ROADS}
        self._last_inference_ms = 0.0

    def update_runtime_settings(
        self,
        confidence_threshold: Optional[float] = None,
        pcu_weights: Optional[Dict[str, float]] = None,
        signal_config: Optional[SignalConfig] = None,
        detect_every_n: Optional[int] = None,
    ) -> None:
        """Apply live parameter updates from the dashboard without restarting streams."""
        if confidence_threshold is not None:
            self.detector.config.confidence_threshold = confidence_threshold
        if pcu_weights is not None:
            self.density_estimator.config.pcu_weights = dict(pcu_weights)
        if signal_config is not None:
            self.signal_controller.config = signal_config
            self.signal_controller.baseline.config = signal_config
        if detect_every_n is not None:
            self.detect_every_n = max(1, detect_every_n)

    def step(self, dt_sec: float = 1.0) -> SimulationStepOutput:
        """Advance the simulation by one frame read and dt_sec of signal time."""
        frames_bgr = self.video_manager.read_all_frames()

        if self.step_index % self.detect_every_n == 0:
            t0 = time.perf_counter()
            self._last_detections = self.detector.detect_batch(frames_bgr)
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0

        sample_shape = next(iter(frames_bgr.values())).shape
        counts_by_road = self.counter.count_all_roads(self._last_detections, sample_shape)
        density_reports = self.density_estimator.update_all(counts_by_road)
        signal_snapshot = self.signal_controller.step(dt_sec, density_reports)

        annotated_frames_rgb: Dict[str, np.ndarray] = {}
        for road in ROADS:
            poly = self.counter.get_roi_polygon(road, sample_shape)
            annotated_bgr = annotate_road_frame(
                frame=frames_bgr[road],
                detections=counts_by_road[road].detections_in_roi,
                roi_polygon=poly,
                road_name=road,
                total_count=counts_by_road[road].total_count,
                signal_state=signal_snapshot.road_states[road],
            )
            annotated_frames_rgb[road] = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        self.step_index += 1
        self.elapsed_sim_sec += dt_sec

        return SimulationStepOutput(
            step_index=self.step_index,
            elapsed_sim_sec=round(self.elapsed_sim_sec, 1),
            annotated_frames_rgb=annotated_frames_rgb,
            counts_by_road=counts_by_road,
            density_reports=density_reports,
            signal_snapshot=signal_snapshot,
            inference_ms=round(self._last_inference_ms, 1),
        )

    def reset(self) -> None:
        """Reset video streams, density EMA history, and signal state."""
        self.video_manager.reset()
        self.density_estimator.reset()
        self.signal_controller.reset()
        self.step_index = 0
        self.elapsed_sim_sec = 0.0
        self._last_detections = {road: [] for road in ROADS}
        self._last_inference_ms = 0.0

    def close(self) -> None:
        """Release underlying video stream resources."""
        self.video_manager.close()

    def __enter__(self) -> "TrafficSimulationEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
