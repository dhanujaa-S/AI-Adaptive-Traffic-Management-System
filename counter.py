"""ROI-based multi-road vehicle counting for Indian urban intersections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import DEFAULT_PCU_WEIGHTS, DEFAULT_ROI_NORMALIZED, ROADS
from src.detector import Detection


@dataclass
class RoadCountResult:
    """Vehicle counts inside a simulated road's approach ROI."""

    road_name: str
    total_count: int
    class_counts: Dict[str, int] = field(default_factory=dict)
    detections_in_roi: List[Detection] = field(default_factory=list)


class RoadVehicleCounter:
    """Filters vehicle detections by road ROI polygon and aggregates class counts."""

    def __init__(
        self,
        roi_normalized: Optional[Dict[str, List[Tuple[float, float]]]] = None,
        tracked_classes: Optional[Sequence[str]] = None,
    ) -> None:
        self.roi_normalized = roi_normalized or dict(DEFAULT_ROI_NORMALIZED)
        self.tracked_classes = tuple(tracked_classes or DEFAULT_PCU_WEIGHTS.keys())

    def get_roi_polygon(self, road_name: str, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """Convert normalized ROI points to pixel coordinates for the given frame shape."""
        height, width = frame_shape[0], frame_shape[1]
        norm_pts = self.roi_normalized.get(road_name, DEFAULT_ROI_NORMALIZED["North"])
        pixel_pts = [[int(round(x * width)), int(round(y * height))] for x, y in norm_pts]
        return np.array(pixel_pts, dtype=np.int32)

    @staticmethod
    def is_inside_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """Return True if the vehicle anchor point lies inside or on the ROI polygon."""
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def count_road(
        self,
        road_name: str,
        detections: Sequence[Detection],
        frame_shape: Tuple[int, ...],
    ) -> RoadCountResult:
        """Count vehicles per class inside the specified road's ROI."""
        polygon = self.get_roi_polygon(road_name, frame_shape)
        class_counts: Dict[str, int] = {cls_name: 0 for cls_name in self.tracked_classes}
        in_roi: List[Detection] = []

        for det in detections:
            if self.is_inside_polygon(det.anchor_point, polygon):
                in_roi.append(det)
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        return RoadCountResult(
            road_name=road_name,
            total_count=len(in_roi),
            class_counts=class_counts,
            detections_in_roi=in_roi,
        )

    def count_all_roads(
        self,
        detections_by_road: Dict[str, List[Detection]],
        frame_shape: Tuple[int, ...],
    ) -> Dict[str, RoadCountResult]:
        """Count vehicles across all 4 simulated roads."""
        results: Dict[str, RoadCountResult] = {}
        for road in ROADS:
            road_dets = detections_by_road.get(road, [])
            results[road] = self.count_road(road, road_dets, frame_shape)
        return results
