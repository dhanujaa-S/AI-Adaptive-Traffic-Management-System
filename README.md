# AI-Based Adaptive Traffic Management System (Indian Urban Intersection)

A software-only AI/ML traffic signal control simulation built with **Python**, **YOLOv8**, **OpenCV**, and **Streamlit**, tailored for heterogeneous Indian city traffic using IRC-inspired **Passenger Car Unit (PCU)** density weighting.

## Architecture

- `src/config.py`: Road approaches (`North`, `South`, `East`, `West`), COCO-to-Indian vehicle class mappings, PCU weights, normalized ROI polygons, and timing configs.
- `src/video_manager.py`: Downloads publicly available sample traffic videos and streams synchronized, looping 4-road OpenCV feeds.
- `src/detector.py`: Batched YOLOv8 (`yolov8n.pt`) inference and OpenCV HUD/bounding-box rendering.
- `src/counter.py`: ROI polygon filtering using each vehicle's road-contact anchor point (`bottom-center`).
- `src/density.py`: PCU-weighted road load, EMA smoothing, and congestion classification (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`).
- `src/signal_controller.py`: `GREEN → YELLOW → ALL_RED` adaptive state machine with density-proportional green timing, anti-starvation aging, and a parallel fixed-timer baseline.
- `src/simulation.py`: Orchestration engine combining all 4 modules.
- `app.py`: Interactive Streamlit dashboard.

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run Tests

```bash
python -m pytest -v
```
