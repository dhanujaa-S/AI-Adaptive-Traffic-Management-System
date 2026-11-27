"""End-to-end integration and Streamlit smoke tests for Module 5."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import DEFAULT_FRAME_SIZE, ROADS, SignalConfig
from src.simulation import TrafficSimulationEngine


def test_simulation_engine_end_to_end() -> None:
    """Run multiple simulation steps across all 4 sample videos and verify outputs."""
    cfg = SignalConfig(min_green_sec=5.0, max_green_sec=20.0, yellow_sec=2.0, all_red_sec=1.0)
    with TrafficSimulationEngine(
        signal_config=cfg,
        auto_download_videos=False,
        detect_every_n=2,
    ) as engine:
        outputs = [engine.step(dt_sec=2.0) for _ in range(4)]

        last = outputs[-1]
        assert last.step_index == 4
        assert last.elapsed_sim_sec == 8.0
        assert set(last.annotated_frames_rgb.keys()) == set(ROADS)
        assert set(last.density_reports.keys()) == set(ROADS)

        expected_shape = (DEFAULT_FRAME_SIZE[1], DEFAULT_FRAME_SIZE[0], 3)
        for road in ROADS:
            assert last.annotated_frames_rgb[road].shape == expected_shape
            assert 0.0 <= last.density_reports[road].normalized_density <= 1.0

        # After 8s with min_green=5s, controller should have transitioned beyond initial GREEN
        assert last.signal_snapshot.phase_state in ("GREEN", "YELLOW", "ALL_RED")

        engine.reset()
        assert engine.step_index == 0
        assert engine.elapsed_sim_sec == 0.0


def test_streamlit_dashboard_renders_without_errors() -> None:
    """Smoke test app.py via Streamlit AppTest including a manual Step action."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised exception: {at.exception}"
    assert len(at.title) == 1
    assert "Adaptive Traffic Management" in at.title[0].value

    # Click the 'Step' button in the sidebar and confirm clean rerun
    step_buttons = [btn for btn in at.sidebar.button if "Step" in btn.label]
    assert step_buttons, "Expected 'Step' button in sidebar"
    step_buttons[0].click()
    at.run()
    assert not at.exception, f"AppTest raised exception after Step: {at.exception}"
