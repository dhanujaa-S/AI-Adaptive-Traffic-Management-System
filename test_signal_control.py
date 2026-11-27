"""Unit tests for Module 4: Adaptive Traffic Signal Controller."""

from typing import Dict

from src.config import ROADS, SignalConfig
from src.density import RoadDensityReport
from src.signal_controller import AdaptiveSignalController


def _make_reports(densities: Dict[str, float], capacity: float = 20.0) -> Dict[str, RoadDensityReport]:
    """Helper to build RoadDensityReport dicts for deterministic controller tests."""
    reports: Dict[str, RoadDensityReport] = {}
    for road in ROADS:
        norm = densities.get(road, 0.0)
        pcu = norm * capacity
        reports[road] = RoadDensityReport(
            road_name=road,
            raw_vehicle_count=int(round(pcu)),
            raw_pcu=pcu,
            smoothed_pcu=pcu,
            normalized_density=norm,
            density_percent=round(norm * 100.0, 1),
            congestion_level="LOW" if norm < 0.3 else "HIGH",
            class_counts={"car": int(round(pcu))},
        )
    return reports


def test_green_duration_scales_with_density() -> None:
    """Verify busier roads receive longer green times within configured bounds."""
    cfg = SignalConfig(min_green_sec=10.0, max_green_sec=45.0)
    ctrl = AdaptiveSignalController(cfg)

    reports = _make_reports({"North": 0.9, "South": 0.1, "East": 0.2, "West": 0.1})
    north_green = ctrl.compute_green_duration("North", reports)
    south_green = ctrl.compute_green_duration("South", reports)

    assert cfg.min_green_sec <= south_green < north_green <= cfg.max_green_sec


def test_phase_fsm_transitions_green_yellow_all_red() -> None:
    """Verify clean state transitions GREEN -> YELLOW -> ALL_RED -> next GREEN."""
    cfg = SignalConfig(min_green_sec=10.0, max_green_sec=30.0, yellow_sec=3.0, all_red_sec=1.0)
    ctrl = AdaptiveSignalController(cfg)
    reports = _make_reports({"North": 0.2, "South": 0.8, "East": 0.1, "West": 0.1})

    # Initial phase is North GREEN (10s)
    snap = ctrl.step(10.0, reports)
    assert snap.active_road == "North"
    assert snap.phase_state == "YELLOW"
    assert snap.road_states["North"] == "YELLOW"

    snap = ctrl.step(3.0, reports)
    assert snap.phase_state == "ALL_RED"
    assert all(state == "RED" for state in snap.road_states.values())

    snap = ctrl.step(1.0, reports)
    assert snap.phase_state == "GREEN"
    assert snap.active_road == "South"  # Highest density candidate
    assert snap.road_states["South"] == "GREEN"
    assert snap.road_states["North"] == "RED"


def test_starvation_prevention_serves_waiting_low_density_road() -> None:
    """Verify a low-traffic road is eventually served and never starves indefinitely."""
    cfg = SignalConfig(
        min_green_sec=10.0,
        max_green_sec=20.0,
        yellow_sec=2.0,
        all_red_sec=1.0,
        max_wait_sec=40.0,
    )
    ctrl = AdaptiveSignalController(cfg)
    reports = _make_reports({"North": 0.9, "South": 0.9, "East": 0.05, "West": 0.05})

    # Worst-case bound when multiple low-density roads cross max_wait_sec during a max_green phase
    worst_case_bound = (
        cfg.max_wait_sec
        + cfg.max_green_sec
        + cfg.min_green_sec
        + 2 * (cfg.yellow_sec + cfg.all_red_sec)
        + 2.0
    )
    served_roads = set()
    for _ in range(120):
        snap = ctrl.step(1.0, reports)
        if snap.phase_state == "GREEN":
            served_roads.add(snap.active_road)
        assert all(w <= worst_case_bound for w in snap.wait_times_sec.values())

    assert "East" in served_roads
    assert "West" in served_roads


def test_adaptive_outperforms_fixed_timer_under_asymmetric_load() -> None:
    """Verify adaptive controller achieves lower PCU delay than fixed-timer baseline."""
    cfg = SignalConfig(min_green_sec=10.0, max_green_sec=40.0, yellow_sec=3.0, all_red_sec=1.0, fixed_green_sec=25.0)
    ctrl = AdaptiveSignalController(cfg)

    # Heavy East-West traffic, almost empty North-South
    reports = _make_reports({"North": 0.05, "South": 0.05, "East": 0.95, "West": 0.85})
    snap = ctrl.snapshot()
    for _ in range(200):
        snap = ctrl.step(1.0, reports)

    assert snap.adaptive_delay_pcu_sec < snap.fixed_delay_pcu_sec
    assert snap.delay_reduction_percent > 0.0
