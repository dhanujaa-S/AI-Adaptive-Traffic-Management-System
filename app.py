"""Streamlit dashboard for the AI-Based Adaptive Traffic Management System (India)."""

from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.config import DEFAULT_PCU_WEIGHTS, ROADS, DetectorConfig, SignalConfig
from src.detector import VehicleDetector
from src.simulation import SimulationStepOutput, TrafficSimulationEngine

st.set_page_config(
    page_title="AI Adaptive Traffic Management System — India",
    page_icon="🚦",
    layout="wide",
)

SIGNAL_BADGE = {
    "GREEN": "🟢 GREEN",
    "YELLOW": "🟡 YELLOW",
    "RED": "🔴 RED",
}

CONGESTION_COLOR = {
    "LOW": "normal",
    "MODERATE": "off",
    "HIGH": "inverse",
    "CRITICAL": "inverse",
}


@st.cache_resource
def load_shared_detector() -> VehicleDetector:
    """Cache YOLOv8 model weights once across Streamlit reruns."""
    return VehicleDetector(DetectorConfig())


def init_session_state() -> None:
    """Initialize simulation engine and UI history buffers in session_state."""
    if "engine" not in st.session_state:
        detector = load_shared_detector()
        st.session_state.engine = TrafficSimulationEngine(
            detector=detector,
            auto_download_videos=True,
            detect_every_n=2,
        )
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "last_output" not in st.session_state:
        # Run 1 initial frame so the dashboard immediately renders live detections
        st.session_state.last_output = st.session_state.engine.step(dt_sec=1.0)
    if "history" not in st.session_state:
        st.session_state.history = []
        _record_history(st.session_state.last_output)


def _record_history(output: SimulationStepOutput) -> None:
    row: Dict[str, float] = {
        "time_sec": output.elapsed_sim_sec,
        "Adaptive Delay (PCU·s)": output.signal_snapshot.adaptive_delay_pcu_sec,
        "Fixed-Timer Delay (PCU·s)": output.signal_snapshot.fixed_delay_pcu_sec,
    }
    for road in ROADS:
        row[f"{road} Density %"] = output.density_reports[road].density_percent
    st.session_state.history.append(row)
    if len(st.session_state.history) > 120:
        st.session_state.history = st.session_state.history[-120:]


def render_sidebar() -> tuple[float, int]:
    """Render sidebar controls and apply live parameter updates."""
    st.sidebar.header("🎛️ Simulation Controls")

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("▶ Start" if not st.session_state.is_running else "⏸ Pause", use_container_width=True):
        st.session_state.is_running = not st.session_state.is_running
        st.rerun()

    if col_b.button("⏭ Step", use_container_width=True, disabled=st.session_state.is_running):
        out = st.session_state.engine.step(dt_sec=st.session_state.get("dt_sec", 1.0))
        st.session_state.last_output = out
        _record_history(out)
        st.rerun()

    if st.sidebar.button("🔄 Reset Intersection", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.engine.reset()
        out = st.session_state.engine.step(dt_sec=1.0)
        st.session_state.last_output = out
        st.session_state.history = []
        _record_history(out)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("⚡ Vision & Timing Settings")
    confidence = st.sidebar.slider("YOLOv8 Confidence", 0.15, 0.85, 0.35, 0.05)
    detect_every_n = st.sidebar.slider("Detect Every N Frames (CPU Tuning)", 1, 5, 2, 1)
    dt_sec = st.sidebar.slider("Simulated Seconds / Step", 0.5, 3.0, 1.0, 0.5, key="dt_sec")

    min_green = st.sidebar.slider("Min Green Time (s)", 5.0, 20.0, 10.0, 1.0)
    max_green = st.sidebar.slider("Max Green Time (s)", 20.0, 90.0, 45.0, 5.0)
    yellow_sec = st.sidebar.slider("Yellow Clearance (s)", 2.0, 5.0, 3.0, 0.5)
    max_wait = st.sidebar.slider("Max Wait / Starvation Guard (s)", 30.0, 150.0, 90.0, 5.0)
    fixed_green = st.sidebar.slider("Baseline Fixed Green (s)", 10.0, 60.0, 25.0, 5.0)

    with st.sidebar.expander("🇮🇳 Indian Traffic PCU Weights"):
        pcu_weights = {}
        for cls_name, default_val in DEFAULT_PCU_WEIGHTS.items():
            pcu_weights[cls_name] = st.number_input(
                f"{cls_name.replace('_', ' ').title()} PCU",
                min_value=0.1,
                max_value=5.0,
                value=float(default_val),
                step=0.25,
            )

    signal_cfg = SignalConfig(
        min_green_sec=min_green,
        max_green_sec=max(min_green + 5.0, max_green),
        yellow_sec=yellow_sec,
        all_red_sec=1.0,
        max_wait_sec=max_wait,
        fixed_green_sec=fixed_green,
    )
    st.session_state.engine.update_runtime_settings(
        confidence_threshold=confidence,
        pcu_weights=pcu_weights,
        signal_config=signal_cfg,
        detect_every_n=detect_every_n,
    )
    return dt_sec, detect_every_n


def render_dashboard(output: SimulationStepOutput) -> None:
    """Render KPI metrics, 4-road feeds, signal status matrix, and analytics charts."""
    snap = output.signal_snapshot
    total_vehicles = sum(c.total_count for c in output.counts_by_road.values())
    total_pcu = sum(r.smoothed_pcu for r in output.density_reports.values())

    st.title("🚦 AI-Based Adaptive Traffic Management System (Indian Urban Intersection)")
    st.caption(
        f"Step #{output.step_index} | Sim Time: {output.elapsed_sim_sec:.1f}s | "
        f"YOLOv8n Batch Inference: {output.inference_ms:.1f} ms | Software-Only Simulation"
    )

    # Top KPI Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Active Road", f"{snap.active_road}", f"Phase: {snap.phase_state}")
    k2.metric("Phase Countdown", f"{snap.remaining_sec:.1f} s", f"Alloc Green: {snap.allocated_green_sec:.0f} s")
    k3.metric("Vehicles in ROIs", f"{total_vehicles}", f"Cycle #{snap.cycle_count}")
    k4.metric("Total PCU Load", f"{total_pcu:.1f} PCU", "IRC-weighted")
    k5.metric(
        "Delay vs Fixed Timer",
        f"{snap.delay_reduction_percent:+.1f}%",
        f"Adapt {snap.adaptive_delay_pcu_sec:.0f} vs Fixed {snap.fixed_delay_pcu_sec:.0f}",
    )

    st.info(f"**Adaptive Controller Decision:** {snap.decision_reason}  |  *(Fixed-Timer Baseline currently serving: {snap.fixed_active_road} — {snap.fixed_phase_state})*")

    # 2x2 Video Feeds + Intersection Status Matrix
    left_col, right_col = st.columns([1.45, 1.0])

    with left_col:
        st.subheader("📹 Live 4-Approach Computer Vision Feeds")
        top_row = st.columns(2)
        bot_row = st.columns(2)
        grid_slots = {
            "North": top_row[0],
            "South": top_row[1],
            "East": bot_row[0],
            "West": bot_row[1],
        }
        for road, slot in grid_slots.items():
            report = output.density_reports[road]
            sig = snap.road_states[road]
            with slot:
                st.markdown(
                    f"**{road}** — {SIGNAL_BADGE[sig]} | "
                    f"`{report.raw_vehicle_count} veh` · `{report.smoothed_pcu:.1f} PCU` · "
                    f"**{report.density_percent:.0f}%** ({report.congestion_level})"
                )
                st.image(output.annotated_frames_rgb[road], use_container_width=True)

    with right_col:
        st.subheader("🚦 4-Way Signal & Density Overview")
        status_rows: List[Dict[str, object]] = []
        for road in ROADS:
            rep = output.density_reports[road]
            status_rows.append(
                {
                    "Road": road,
                    "Signal": SIGNAL_BADGE[snap.road_states[road]],
                    "Vehicles": rep.raw_vehicle_count,
                    "PCU Load": rep.smoothed_pcu,
                    "Density %": rep.density_percent,
                    "Congestion": rep.congestion_level,
                    "Wait (s)": snap.wait_times_sec[road],
                }
            )
        st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)

        st.subheader("📊 Density (%) by Approach")
        density_df = pd.DataFrame(
            {
                "Road": list(ROADS),
                "Density %": [output.density_reports[r].density_percent for r in ROADS],
            }
        ).set_index("Road")
        st.bar_chart(density_df, height=210)

        st.subheader("🛺 Vehicle Class Mix Across Roads")
        class_rows = []
        for road in ROADS:
            row = {"Road": road}
            row.update(output.counts_by_road[road].class_counts)
            class_rows.append(row)
        class_df = pd.DataFrame(class_rows).set_index("Road")
        st.bar_chart(class_df, height=220)

    # Bottom Rolling Time-Series Charts
    if len(st.session_state.history) > 1:
        hist_df = pd.DataFrame(st.session_state.history).set_index("time_sec")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 Live Traffic Density Trend (%)")
            density_cols = [f"{road} Density %" for road in ROADS]
            st.line_chart(hist_df[density_cols], height=230)
        with c2:
            st.subheader("⏱️ Cumulative Red-Light Delay: Adaptive vs. Fixed Timer")
            st.line_chart(
                hist_df[["Adaptive Delay (PCU·s)", "Fixed-Timer Delay (PCU·s)"]],
                height=230,
            )


def main() -> None:
    init_session_state()
    dt_sec, _ = render_sidebar()
    render_dashboard(st.session_state.last_output)

    if st.session_state.is_running:
        time.sleep(0.15)
        next_out = st.session_state.engine.step(dt_sec=dt_sec)
        st.session_state.last_output = next_out
        _record_history(next_out)
        st.rerun()


if __name__ == "__main__":
    main()
