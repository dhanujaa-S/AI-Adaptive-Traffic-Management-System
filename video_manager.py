"""Video ingestion and sample traffic video management for 4 simulated roads."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.config import DEFAULT_FRAME_SIZE, ROADS, SAMPLE_VIDEO_SOURCES, VIDEO_DIR


def download_file(url: str, destination: Path, timeout_sec: int = 30) -> Path:
    """Download a file from a public URL to destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AdaptiveTrafficIndia/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response, open(tmp_path, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp_path.replace(destination)
    return destination


def ensure_sample_videos(
    video_dir: Path = VIDEO_DIR,
    sources: Dict[str, Dict[str, object]] = SAMPLE_VIDEO_SOURCES,
    force: bool = False,
) -> Dict[str, Path]:
    """Ensure sample traffic videos exist locally for all configured roads.

    Deduplicates downloads when multiple roads share the same public URL.
    """
    video_dir.mkdir(parents=True, exist_ok=True)
    url_cache: Dict[str, Path] = {}
    road_paths: Dict[str, Path] = {}

    for road, info in sources.items():
        filename = str(info["filename"])
        url = str(info["url"])
        dest = video_dir / filename

        if dest.exists() and dest.stat().st_size > 0 and not force:
            url_cache.setdefault(url, dest)
            road_paths[road] = dest
            continue

        if url in url_cache and url_cache[url].exists():
            shutil.copy2(url_cache[url], dest)
        else:
            download_file(url, dest)
            url_cache[url] = dest

        road_paths[road] = dest

    return road_paths


class RoadVideoStream:
    """Looping OpenCV video stream for a single simulated road approach."""

    def __init__(
        self,
        road_name: str,
        video_path: Path,
        frame_size: Tuple[int, int] = DEFAULT_FRAME_SIZE,
        start_offset_ratio: float = 0.0,
    ) -> None:
        self.road_name = road_name
        self.video_path = Path(video_path)
        self.frame_size = frame_size
        self.start_offset_ratio = max(0.0, min(0.95, start_offset_ratio))

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found for {road_name}: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video stream for {road_name}: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if self.fps <= 1.0:
            self.fps = 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_index = 0
        self._seek_initial_offset()

    def _seek_initial_offset(self) -> None:
        if self.total_frames > 0 and self.start_offset_ratio > 0.0:
            start_frame = int(self.total_frames * self.start_offset_ratio)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.frame_index = start_frame

    def read_frame(self) -> np.ndarray:
        """Read the next resized BGR frame, looping seamlessly at EOF."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Unable to read frame from {self.video_path}")

        self.frame_index += 1
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        return frame

    def reset(self) -> None:
        """Reset stream back to its initial offset."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_index = 0
        self._seek_initial_offset()

    def close(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self.cap is not None:
            self.cap.release()


class MultiRoadVideoManager:
    """Manages synchronized video streams for the 4 simulated intersection roads."""

    def __init__(
        self,
        video_paths: Optional[Dict[str, Path]] = None,
        frame_size: Tuple[int, int] = DEFAULT_FRAME_SIZE,
        auto_download: bool = True,
    ) -> None:
        self.frame_size = frame_size
        if video_paths is None:
            if auto_download:
                video_paths = ensure_sample_videos()
            else:
                video_paths = {
                    road: VIDEO_DIR / str(SAMPLE_VIDEO_SOURCES[road]["filename"])
                    for road in ROADS
                }

        self.streams: Dict[str, RoadVideoStream] = {}
        for road in ROADS:
            if road not in video_paths:
                raise ValueError(f"Missing video path for road '{road}'")
            offset = float(SAMPLE_VIDEO_SOURCES.get(road, {}).get("start_offset_ratio", 0.0))
            self.streams[road] = RoadVideoStream(
                road_name=road,
                video_path=video_paths[road],
                frame_size=self.frame_size,
                start_offset_ratio=offset,
            )

    def read_all_frames(self) -> Dict[str, np.ndarray]:
        """Read one resized BGR frame from each of the 4 simulated roads."""
        return {road: stream.read_frame() for road, stream in self.streams.items()}

    def get_metadata(self) -> Dict[str, Dict[str, float]]:
        """Return FPS and total frame count per road."""
        return {
            road: {
                "fps": stream.fps,
                "total_frames": float(stream.total_frames),
                "frame_index": float(stream.frame_index),
            }
            for road, stream in self.streams.items()
        }

    def reset(self) -> None:
        """Reset all 4 streams."""
        for stream in self.streams.values():
            stream.reset()

    def close(self) -> None:
        """Close all 4 streams."""
        for stream in self.streams.values():
            stream.close()

    def __enter__(self) -> "MultiRoadVideoManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
