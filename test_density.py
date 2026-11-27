"""Unit tests for Module 3: PCU-based traffic density estimator."""

import pytest

from src.config import ROADS, DensityConfig
from src.counter import RoadCountResult
from src.density import TrafficDensityEstimator


def test_pcu_weighting_reflects_indian_vehicle_mix() -> None:
    """Verify buses/trucks carry higher PCU load than motorcycles and cars."""
    estimator = TrafficDensityEstimator(
        DensityConfig(road_capacity_pcu={road: 20.0 for road in ROADS}, ema_alpha=1.0)
    )

    two_wheeler_heavy = RoadCountResult(
        road_name="North",
        total_count=10,
        class_counts={"motorcycle": 10},  # 10 * 0.5 = 5.0 PCU
    )
    bus_heavy = RoadCountResult(
        road_name="South",
        total_count=4,
        class_counts={"bus": 3, "car": 1},  # 3 * 3.0 + 1 * 1.0 = 10.0 PCU
    )

    reports = estimator.update_all({"North": two_wheeler_heavy, "South": bus_heavy})
    assert reports["North"].raw_pcu == pytest.approx(5.0)
    assert reports["South"].raw_pcu == pytest.approx(10.0)
    assert reports["South"].normalized_density > reports["North"].normalized_density
    assert reports["North"].congestion_level == "LOW"
    assert reports["South"].congestion_level == "MODERATE"


def test_empty_and_saturated_roads_clamp_cleanly() -> None:
    """Verify zero traffic and overflow traffic clamp within [0.0, 1.0]."""
    estimator = TrafficDensityEstimator(
        DensityConfig(road_capacity_pcu={road: 10.0 for road in ROADS}, ema_alpha=1.0)
    )

    empty = RoadCountResult(road_name="East", total_count=0, class_counts={})
    gridlock = RoadCountResult(
        road_name="West",
        total_count=8,
        class_counts={"truck": 5, "bus": 3},  # 24.0 PCU > capacity 10.0
    )

    reports = estimator.update_all({"East": empty, "West": gridlock})
    assert reports["East"].normalized_density == 0.0
    assert reports["East"].congestion_level == "LOW"
    assert reports["West"].normalized_density == 1.0
    assert reports["West"].density_percent == 100.0
    assert reports["West"].congestion_level == "CRITICAL"


def test_ema_smoothing_dampens_single_frame_spikes() -> None:
    """Verify EMA prevents sudden single-frame detection spikes from jolting density."""
    estimator = TrafficDensityEstimator(
        DensityConfig(road_capacity_pcu={road: 20.0 for road in ROADS}, ema_alpha=0.25)
    )

    baseline = RoadCountResult(road_name="North", total_count=4, class_counts={"car": 4})  # 4 PCU
    spike = RoadCountResult(road_name="North", total_count=16, class_counts={"car": 16})  # 16 PCU

    r1 = estimator.update_road(baseline)
    assert r1.smoothed_pcu == pytest.approx(4.0)

    r2 = estimator.update_road(spike)
    # 0.25 * 16 + 0.75 * 4 = 7.0
    assert r2.smoothed_pcu == pytest.approx(7.0)
    assert r2.smoothed_pcu < r2.raw_pcu

    estimator.reset()
    r3 = estimator.update_road(spike)
    assert r3.smoothed_pcu == pytest.approx(16.0)
