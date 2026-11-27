"""PCU-based traffic density calculation and temporal smoothing for Indian roads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.config import ROADS, DensityConfig
from src.counter import RoadCountResult


@dataclass(frozen=True)
class RoadDensityReport:
    """Computed traffic density metrics for a single simulated road approach."""

    road_name: str
    raw_vehicle_count: int
    raw_pcu: float
    smoothed_pcu: float
    normalized_density: float  # Clamped to [0.0, 1.0]
    density_percent: float  # 0.0 to 100.0
    congestion_level: str  # LOW | MODERATE | HIGH | CRITICAL
    class_counts: Dict[str, int] = field(default_factory=dict)


class TrafficDensityEstimator:
    """Estimates road congestion using IRC-inspired PCU weights and EMA smoothing."""

    def __init__(self, config: Optional[DensityConfig] = None) -> None:
        self.config = config or DensityConfig()
        if not 0.0 < self.config.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0.0, 1.0]")
        self._smoothed_pcu: Dict[str, Optional[float]] = {road: None for road in ROADS}

    def compute_raw_pcu(self, class_counts: Dict[str, int]) -> float:
        """Convert heterogeneous vehicle class counts into Passenger Car Units (PCU)."""
        total_pcu = 0.0
        for cls_name, count in class_counts.items():
            if count <= 0:
                continue
            weight = float(self.config.pcu_weights.get(cls_name, 1.0))
            total_pcu += count * weight
        return total_pcu

    @staticmethod
    def classify_congestion(normalized_density: float) -> str:
        """Map normalized density [0.0, 1.0] to a human-readable congestion level."""
        if normalized_density < 0.30:
            return "LOW"
        if normalized_density < 0.60:
            return "MODERATE"
        if normalized_density < 0.85:
            return "HIGH"
        return "CRITICAL"

    def update_road(self, count_result: RoadCountResult) -> RoadDensityReport:
        """Update EMA-smoothed PCU density for a single road."""
        road = count_result.road_name
        raw_pcu = self.compute_raw_pcu(count_result.class_counts)
        prev = self._smoothed_pcu.get(road)

        if prev is None:
            smoothed_pcu = raw_pcu
        else:
            alpha = self.config.ema_alpha
            smoothed_pcu = alpha * raw_pcu + (1.0 - alpha) * prev

        self._smoothed_pcu[road] = smoothed_pcu

        capacity = max(1e-6, float(self.config.road_capacity_pcu.get(road, 18.0)))
        normalized = min(1.0, max(0.0, smoothed_pcu / capacity))

        return RoadDensityReport(
            road_name=road,
            raw_vehicle_count=count_result.total_count,
            raw_pcu=round(raw_pcu, 2),
            smoothed_pcu=round(smoothed_pcu, 2),
            normalized_density=round(normalized, 4),
            density_percent=round(normalized * 100.0, 1),
            congestion_level=self.classify_congestion(normalized),
            class_counts=dict(count_result.class_counts),
        )

    def update_all(
        self, counts_by_road: Dict[str, RoadCountResult]
    ) -> Dict[str, RoadDensityReport]:
        """Update and return density reports across all 4 simulated roads."""
        reports: Dict[str, RoadDensityReport] = {}
        for road in ROADS:
            count_result = counts_by_road.get(
                road,
                RoadCountResult(road_name=road, total_count=0, class_counts={}),
            )
            reports[road] = self.update_road(count_result)
        return reports

    def reset(self) -> None:
        """Clear EMA history across all roads."""
        self._smoothed_pcu = {road: None for road in ROADS}
