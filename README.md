# 🚦 AI-Based Adaptive Traffic Signal Management System

An intelligent traffic management system that uses computer vision to monitor traffic at a four-way intersection and automatically adjusts signal timings based on real-time traffic density.

Instead of giving every road the same fixed green signal time, the system identifies which road has more traffic and gives it priority. The goal is to reduce unnecessary waiting time, improve traffic flow, and make signal management smarter.

## 💡 What Does This Project Do?

The system simulates a four-way road intersection using four video streams:

- North Road
- South Road
- East Road
- West Road

It uses **YOLOv8** to detect vehicles in each video, counts them, and estimates traffic density. Based on the detected traffic, an adaptive signal controller decides which road should receive the next green signal and how long it should remain green.

## ✨ Key Features

- **Vehicle Detection:** Detects vehicles using YOLOv8.
- **Multi-Road Monitoring:** Processes four road video streams simultaneously.
- **Vehicle Counting:** Counts vehicles on each road based on their classes.
- **Traffic Density Estimation:** Calculates traffic load using Passenger Car Unit (PCU) weights.
- **Adaptive Signal Timing:** Dynamically adjusts green signal duration according to traffic density.
- **Priority-Based Scheduling:** Gives priority to roads with heavier traffic.
- **Starvation Prevention:** Considers waiting time to prevent a road from being ignored for too long.
- **Performance Comparison:** Compares adaptive signal control with a traditional fixed-timer system.
- **Live Dashboard:** Displays vehicle detection, traffic density, signal status, and performance metrics.

## ⚙️ How It Works

The project follows five main stages:

1. **Video Ingestion:** Reads and synchronizes four road video streams, with frame resizing and FPS control.

2. **Vehicle Detection:** YOLOv8 identifies vehicles in each road feed. ROI (Region of Interest) filtering helps count only vehicles within the selected road area.

3. **Traffic Density Calculation:** Calculates traffic load using vehicle counts and PCU weights. Exponential Moving Average (EMA) smoothing helps avoid sudden fluctuations.

4. **Adaptive Signal Control:** Selects the next road based on traffic priority and waiting time. Green signal duration is adjusted dynamically, followed by yellow and all-red phases.

5. **Visualization:** A Streamlit dashboard displays all four road feeds, traffic density, signal states, timers, and performance metrics.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| YOLOv8 | Vehicle detection |
| OpenCV | Video processing |
| Streamlit | Interactive dashboard |
| NumPy | Numerical computations |
| Plotly | Data visualization |

## 📂 Project Structure

```text
Traffic-Management-System/
│
├── app.py
├── video_manager.py
├── detector.py
├── counter.py
├── density.py
├── signal_controller.py
│
├── videos/
│   ├── North.mp4
│   ├── South.mp4
│   ├── East.mp4
│   └── West.mp4
│
├── requirements.txt
└── README.md
```

## 🚀 How to Run the Project

**1. Clone the repository**

```bash
git clone <your-repository-url>
```

**2. Navigate to the project folder**

```bash
cd Traffic-Management-System
```

**3. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**4. Start the Streamlit application**

Open the project folder in VS Code, open the terminal (CMD), and execute:

```bash
streamlit run app.py
```

**5. Open the dashboard**

Once the application starts, open the local URL displayed in your terminal (usually `http://localhost:8501`).

## 📊 Traffic Signal Logic

The system uses traffic density and waiting time to determine signal priority.

- Higher traffic density increases a road's priority.
- Green signal duration changes dynamically according to traffic load.
- Yellow and all-red phases ensure transitions between signals.
- Waiting-time aging helps prevent starvation.
- A fixed-timer baseline is included for comparison.

## 🎯 Project Goal

The main goal is to demonstrate how computer vision and adaptive decision-making can be combined to improve traditional traffic signal management.

This project is currently a video-based simulation and can serve as a foundation for developing more advanced intelligent transportation systems.

---

**Built with Python, YOLOv8, OpenCV, and Streamlit.**
