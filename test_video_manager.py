"""Unit and integration tests for Module 1: config and video manager."""

from pathlib import Path

import cv2
import numpy as np

from src.config import DEFAULT_FRAME_SIZE, ROADS
from src.video_manager import MultiRoadVideoManager, RoadVideoStream, ensure_sample_videos


def _create_synthetic_mp4(path: Path, num_frames: int = 10) -> Path:
    """Create a small synthetic MP4 video for isolated unit tests."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 240))
    assert writer.isOpened(), "Failed to create synthetic test video"
    for i in range(num_frames):
        frame = np.full((240, 320, 3), fill_value=(i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_road_stream_resizes_and_loops(tmp_path: Path) -> None:
    """Verify RoadVideoStream resizes frames and loops smoothly at EOF."""
    video_file = _create_synthetic_mp4(tmp_path / "synth.mp4", num_frames=5)
    stream = RoadVideoStream("North", video_file, frame_size=DEFAULT_FRAME_SIZE)
    try:
        for _ in range(8):  # Longer than total_frames=5 to verify looping
            frame = stream.read_frame()
            assert frame.shape == (DEFAULT_FRAME_SIZE[1], DEFAULT_FRAME_SIZE[0], 3)
            assert frame.dtype == np.uint8
    finally:
        stream.close()


def test_multi_road_manager_synthetic(tmp_path: Path) -> None:
    """Verify MultiRoadVideoManager reads synchronized frames across all 4 roads."""
    paths = {
        road: _create_synthetic_mp4(tmp_path / f"{road.lower()}.mp4", num_frames=6)
        for road in ROADS
    }
    with MultiRoadVideoManager(video_paths=paths, auto_download=False) as manager:
        frames = manager.read_all_frames()
        assert set(frames.keys()) == set(ROADS)
        for road, frame in frames.items():
            assert frame.shape == (DEFAULT_FRAME_SIZE[1], DEFAULT_FRAME_SIZE[0], 3), road


def test_sample_videos_download_and_open() -> None:
    """Verify public sample traffic videos download cleanly and open in OpenCV."""
    paths = ensure_sample_videos()
    assert set(paths.keys()) == set(ROADS)
    for road, path in paths.items():
        assert path.exists() and path.stat().st_size > 10_000, f"Invalid sample video for {road}"

    with MultiRoadVideoManager(video_paths=paths, auto_download=False) as manager:
        frames = manager.read_all_frames()
        assert set(frames.keys()) == set(ROADS)
        meta = manager.get_metadata()
        for road in ROADS:
            assert frames[road].shape == (DEFAULT_FRAME_SIZE[1], DEFAULT_FRAME_SIZE[0], 3)
            assert meta[road]["total_frames"] > 10
