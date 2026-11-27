"""Adaptive traffic signal controller with starvation prevention and fixed-timer baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.config import ROADS, SignalConfig
from src.density import RoadDensityReport


@dataclass(frozen=True)
class SignalSnapshot:
    """Immutable state snapshot of the 4-way intersection signal controller."""

    road_states: Dict[str, str]  # road -> GREEN | YELLOW | RED
    active_road: str
    phase_state: str  # GREEN | YELLOW | ALL_RED
    remaining_sec: float
    allocated_green_sec: float
    wait_times_sec: Dict[str, float]
    cycle_count: int
    adaptive_delay_pcu_sec: float
    fixed_delay_pcu_sec: float
    delay_reduction_percent: float
    decision_reason: str
    fixed_active_road: str
    fixed_phase_state: str


class FixedTimerBaseline:
    """Round-robin fixed-duration traffic controller for benchmark comparison."""

    def __init__(self, config: SignalConfig) -> None:
        self.config = config
        self.road_index = 0
        self.phase_state = "GREEN"
        self.remaining_sec = float(config.fixed_green_sec)
        self.cumulative_delay_pcu_sec = 0.0

    @property
    def active_road(self) -> str:
        return ROADS[self.road_index]

    def step(self, dt_sec: float, density_reports: Dict[str, RoadDensityReport]) -> None:
        """Advance fixed timer by dt_sec and accumulate delayed PCU-seconds on red roads."""
        if dt_sec <= 0:
            return

        for road in ROADS:
            is_green = self.phase_state == "GREEN" and road == self.active_road
            if not is_green and road in density_reports:
                self.cumulative_delay_pcu_sec += density_reports[road].smoothed_pcu * dt_sec

        remaining_dt = dt_sec
        while remaining_dt > 1e-9:
            if remaining_dt < self.remaining_sec:
                self.remaining_sec -= remaining_dt
                remaining_dt = 0.0
            else:
                remaining_dt -= self.remaining_sec
                if self.phase_state == "GREEN":
                    self.phase_state = "YELLOW"
                    self.remaining_sec = float(self.config.yellow_sec)
                elif self.phase_state == "YELLOW":
                    if self.config.all_red_sec > 0:
                        self.phase_state = "ALL_RED"
                        self.remaining_sec = float(self.config.all_red_sec)
                    else:
                        self._advance_road()
                else:  # ALL_RED
                    self._advance_road()

    def _advance_road(self) -> None:
        self.road_index = (self.road_index + 1) % len(ROADS)
        self.phase_state = "GREEN"
        self.remaining_sec = float(self.config.fixed_green_sec)


class AdaptiveSignalController:
    """Density-proportional 4-way signal FSM with anti-starvation aging."""

    def __init__(self, config: Optional[SignalConfig] = None) -> None:
        self.config = config or SignalConfig()
        self.active_road: str = ROADS[0]
        self.phase_state: str = "GREEN"
        self.allocated_green_sec: float = float(self.config.min_green_sec)
        self.remaining_sec: float = self.allocated_green_sec
        self.wait_times_sec: Dict[str, float] = {road: 0.0 for road in ROADS}
        self.cycle_count: int = 1
        self.cumulative_delay_pcu_sec: float = 0.0
        self.decision_reason: str = "Initial phase: North"
        self.baseline = FixedTimerBaseline(self.config)

    def compute_green_duration(
        self,
        target_road: str,
        density_reports: Dict[str, RoadDensityReport],
    ) -> float:
        """Calculate adaptive green duration bounded by [min_green_sec, max_green_sec]."""
        target_report = density_reports.get(target_road)
        if target_report is None:
            return float(self.config.min_green_sec)

        target_density = target_report.normalized_density
        total_density = sum(r.normalized_density for r in density_reports.values())

        if total_density <= 1e-6:
            return float(self.config.min_green_sec)

        density_share = target_density / total_density
        demand_factor = 0.65 * target_density + 0.35 * density_share
        span = self.config.max_green_sec - self.config.min_green_sec
        duration = self.config.min_green_sec + span * demand_factor
        clamped = max(self.config.min_green_sec, min(self.config.max_green_sec, duration))
        return round(clamped, 1)

    def select_next_road(
        self, density_reports: Dict[str, RoadDensityReport]
    ) -> tuple[str, str]:
        """Choose the next road to receive GREEN and return (road_name, reason)."""
        candidates = [road for road in ROADS if road != self.active_road]

        # 1. Hard anti-starvation guard: serve any road exceeding max_wait_sec
        starved = [
            road for road in candidates if self.wait_times_sec[road] >= self.config.max_wait_sec
        ]
        if starved:
            chosen = max(starved, key=lambda r: self.wait_times_sec[r])
            waited = self.wait_times_sec[chosen]
            return chosen, f"Starvation guard: {chosen} waited {waited:.0f}s"

        # 2. If active road has traffic and all other roads are completely empty, hold active
        other_pcu = sum(
            density_reports[r].smoothed_pcu for r in candidates if r in density_reports
        )
        active_pcu = (
            density_reports[self.active_road].smoothed_pcu
            if self.active_road in density_reports
            else 0.0
        )
        if other_pcu <= 1e-6 and active_pcu > 1e-6:
            return (
                self.active_road,
                f"Hold {self.active_road}: only active approach with traffic",
            )

        # 3. Priority = normalized_density + aging_weight * wait_time
        def priority_score(road: str) -> float:
            density = (
                density_reports[road].normalized_density
                if road in density_reports
                else 0.0
            )
            return density + self.config.aging_weight * self.wait_times_sec[road]

        chosen = max(candidates, key=priority_score)
        chosen_density = (
            density_reports[chosen].density_percent if chosen in density_reports else 0.0
        )
        waited = self.wait_times_sec[chosen]
        return (
            chosen,
            f"Highest priority: {chosen} ({chosen_density:.0f}% density, waited {waited:.0f}s)",
        )

    def _transition_phase(self, density_reports: Dict[str, RoadDensityReport]) -> None:
        if self.phase_state == "GREEN":
            self.phase_state = "YELLOW"
            self.remaining_sec = float(self.config.yellow_sec)
        elif self.phase_state == "YELLOW":
            if self.config.all_red_sec > 0:
                self.phase_state = "ALL_RED"
                self.remaining_sec = float(self.config.all_red_sec)
            else:
                self._start_next_green(density_reports)
        else:  # ALL_RED
            self._start_next_green(density_reports)

    def _start_next_green(self, density_reports: Dict[str, RoadDensityReport]) -> None:
        next_road, reason = self.select_next_road(density_reports)
        self.active_road = next_road
        self.wait_times_sec[next_road] = 0.0
        self.phase_state = "GREEN"
        self.allocated_green_sec = self.compute_green_duration(next_road, density_reports)
        self.remaining_sec = self.allocated_green_sec
        self.cycle_count += 1
        self.decision_reason = reason

    def step(
        self, dt_sec: float, density_reports: Dict[str, RoadDensityReport]
    ) -> SignalSnapshot:
        """Advance both adaptive controller and fixed baseline by dt_sec."""
        if dt_sec > 0:
            # Update wait times and delayed PCU-seconds
            for road in ROADS:
                is_green = self.phase_state == "GREEN" and road == self.active_road
                if is_green:
                    self.wait_times_sec[road] = 0.0
                else:
                    self.wait_times_sec[road] += dt_sec
                    if road in density_reports:
                        self.cumulative_delay_pcu_sec += (
                            density_reports[road].smoothed_pcu * dt_sec
                        )

            self.baseline.step(dt_sec, density_reports)

            remaining_dt = dt_sec
            while remaining_dt > 1e-9:
                if remaining_dt < self.remaining_sec:
                    self.remaining_sec -= remaining_dt
                    remaining_dt = 0.0
                else:
                    remaining_dt -= self.remaining_sec
                    self._transition_phase(density_reports)

        return self.snapshot()

    def snapshot(self) -> SignalSnapshot:
        """Return current immutable state snapshot."""
        road_states: Dict[str, str] = {}
        for road in ROADS:
            if road == self.active_road and self.phase_state in ("GREEN", "YELLOW"):
                road_states[road] = self.phase_state
            else:
                road_states[road] = "RED"

        fixed_delay = self.baseline.cumulative_delay_pcu_sec
        adaptive_delay = self.cumulative_delay_pcu_sec
        if fixed_delay > 1e-6:
            reduction = ((fixed_delay - adaptive_delay) / fixed_delay) * 100.0
        else:
            reduction = 0.0

        return SignalSnapshot(
            road_states=road_states,
            active_road=self.active_road,
            phase_state=self.phase_state,
            remaining_sec=round(self.remaining_sec, 1),
            allocated_green_sec=round(self.allocated_green_sec, 1),
            wait_times_sec={r: round(v, 1) for r, v in self.wait_times_sec.items()},
            cycle_count=self.cycle_count,
            adaptive_delay_pcu_sec=round(adaptive_delay, 2),
            fixed_delay_pcu_sec=round(fixed_delay, 2),
            delay_reduction_percent=round(reduction, 1),
            decision_reason=self.decision_reason,
            fixed_active_road=self.baseline.active_road,
            fixed_phase_state=self.baseline.phase_state,
        )

    def reset(self) -> None:
        """Reset signal state and baseline comparison counters."""
        self.__init__(self.config)
