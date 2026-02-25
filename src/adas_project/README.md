# Adaptive Intent-Aware ADAS (Advanced Driver Assistance System)

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/index.html)
[![Gazebo](https://img.shields.io/badge/Simulation-Gazebo-orange.svg)](https://gazebosim.org/)
[![Deep Learning](https://img.shields.io/badge/AI-PyTorch-red.svg)](https://pytorch.org/)

A research-oriented ADAS framework that combines classical safety metrics with deep-learning-based driver intent prediction. This system dynamically adjusts safety thresholds (TTC) based on predicted traffic behaviors, minimizing nuisance alerts while maintaining peak safety.

## 📌 Architecture Overview

The system follows a modular ROS2-based architecture, bridging high-fidelity simulation with real-time inference.

```mermaid
graph TD
    subgraph "Simulation (Gazebo)"
        V[Ego Vehicle] --> S[LiDAR / Camera / Odometry]
        T[Traffic Actors] --> S
    end

    subgraph "Perception Layer"
        S --> INF[Inference Node (GRU Model)]
        S --> CAD[Classical ADAS Node]
    end

    subgraph "Intelligence Layer"
        INF -- "Intent Probabilities" --> MAD[ML-Adaptive ADAS]
        CAD -- "Fixed Threshold Alerts" --> ARB[Control Arbitration]
        MAD -- "Dynamic Threshold Alerts" --> ARB
    end

    subgraph "Control Layer"
        ARB -- "Safe cmd_vel" --> V
        U[User Keyboard] -- "Requested Motion" --> ARB
    end

    click INF "scripts/inference_node.py"
    click MAD "scripts/ml_adas.py"
    click ARB "scripts/control_arbitration_node.py"
```

## 🚀 Key Features

- **Hybrid Safety Logic**: Operates both a deterministic classical ADAS and a predictive ML-driven ADAS.
- **Intent-Aware Adaptation**: Uses a **GRU (Gated Recurrent Unit)** network to classify surrounding vehicle behaviors (Aggressive, Defensive, etc.) and adjusts safety margins accordingly.
- **Hysteresis-Aware FSM**: A robust Finite State Machine manages transitions between `MANUAL`, `WARNING`, `ASSIST`, and `EMERGENCY_BRAKE` states to prevent chatter.
- **Integrated HUD**: High-definition OpenCV-based dashboard with real-time telemetry, TTC calculation, and safety status overlays.
- **Automated Evaluation**: Generates ROC curves, confusion matrices, and safety exposure reports (crashes vs close calls) automatically after each run.

## 🛠 Getting Started

### Prerequisites
- ROS2 Humble
- Gazebo Sim
- Python 3.10+
- PyTorch, NumPy, Pandas, Matplotlib, Seaborn, OpenCV

### Installation
```bash
# Clone the repository
mkdir -p ~/adas_ws/src
cd ~/adas_ws/src
git clone <repository_url> adas_project

# Install dependencies
rosdep install --from-paths . --ignore-src -r -y

# Build the workspace
cd ~/adas_ws
colcon build --symlink-install
source install/setup.bash
```

### Running the System
**Manual Mode (Traditional):**
```bash
./src/adas_project/scripts/auto_fixed.sh
```

**ML-Adaptive Mode:**
```bash
./src/adas_project/scripts/auto_ml.sh
```

## 📊 Evaluation & Metrics
The system logs every run to the `evaluation/` directory.
- `performance_roc.png`: Sensitivity vs. False Positive comparison.
- `adas_performance_raw.csv`: High-frequency telemetry log.
- `safety_exposure_report.txt`: Cumulative time spent in dangerous zones.

## 📂 Project Structure
- `scripts/`: Implementation of all ROS2 nodes and training pipelines.
- `launch/`: Multi-node orchestration files.
- `urdf/`: Vehicle and sensor definitions.
- `models/`: Trained intent prediction weights (`.pt`).
- `worlds/`: Simulated highway and traffic environments.

---
**Developed for the ADAS Minor Research Project.**
