"""YOLOv8 vehicle detection wrapper and OpenCV frame annotation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import CLASS_COLORS_BGR, MODELS_DIR, DetectorConfig

# Keep Ultralytics settings self-contained inside the project directory
os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS_DIR / ".ultralytics"))
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """Single detected vehicle in a traffic video frame."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    anchor_point: Tuple[int, int]  # Bottom-center point on the road plane


class VehicleDetector:
    """Detects Indian urban vehicle classes using YOLOv8."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        model_path = Path(self.config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(model_path))
        self.target_class_ids = sorted(self.config.class_map.keys())

    def _parse_result(self, result) -> List[Detection]:
        detections: List[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy_list = boxes.xyxy.cpu().numpy()
        cls_list = boxes.cls.cpu().numpy().astype(int)
        conf_list = boxes.conf.cpu().numpy()

        for xyxy, cls_id, conf in zip(xyxy_list, cls_list, conf_list):
            class_name = self.config.class_map.get(int(cls_id))
            if class_name is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            anchor = ((x1 + x2) // 2, y2)
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    anchor_point=anchor,
                )
            )
        return detections

    def detect_frame(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Run YOLOv8 inference on a single BGR frame."""
        conf = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.confidence_threshold
        )
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            classes=self.target_class_ids,
            verbose=False,
        )
        if not results:
            return []
        return self._parse_result(results[0])

    def detect_batch(
        self,
        frames_by_road: Dict[str, np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, List[Detection]]:
        """Run batched YOLOv8 inference across all 4 simulated road frames."""
        if not frames_by_road:
            return {}
        conf = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.confidence_threshold
        )
        roads = list(frames_by_road.keys())
        frame_list = [frames_by_road[road] for road in roads]
        results = self.model.predict(
            source=frame_list,
            conf=conf,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            classes=self.target_class_ids,
            verbose=False,
        )
        return {
            road: self._parse_result(res)
            for road, res in zip(roads, results)
        }


def annotate_road_frame(
    frame: np.ndarray,
    detections: Sequence[Detection],
    roi_polygon: Optional[np.ndarray] = None,
    road_name: Optional[str] = None,
    total_count: Optional[int] = None,
    signal_state: Optional[str] = None,
) -> np.ndarray:
    """Draw ROI zone, bounding boxes, class labels, and HUD badge on a frame."""
    annotated = frame.copy()

    if roi_polygon is not None and len(roi_polygon) >= 3:
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [roi_polygon], (255, 190, 0))
        cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)
        cv2.polylines(annotated, [roi_polygon], isClosed=True, color=(255, 200, 0), thickness=2)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS_BGR.get(det.class_name, (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, det.anchor_point, 3, color, -1)

        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y1, th + 6)
        cv2.rectangle(
            annotated,
            (x1, label_y - th - 6),
            (x1 + tw + 4, label_y + baseline - 2),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Top HUD bar
    hud_parts = []
    if road_name:
        hud_parts.append(road_name.upper())
    if total_count is not None:
        hud_parts.append(f"Vehicles: {total_count}")
    if signal_state:
        hud_parts.append(f"Signal: {signal_state}")

    if hud_parts:
        hud_text = " | ".join(hud_parts)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (20, 20, 20), -1)
        signal_color = (255, 255, 255)
        if signal_state == "GREEN":
            signal_color = (80, 255, 80)
        elif signal_state == "YELLOW":
            signal_color = (0, 220, 255)
        elif signal_state == "RED":
            signal_color = (80, 80, 255)
        cv2.putText(
            annotated,
            hud_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            signal_color,
            2,
            cv2.LINE_AA,
        )

    return annotated
