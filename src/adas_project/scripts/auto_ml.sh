#!/bin/bash
# auto_ml.sh — Autonomous driving with ML-adaptive ADAS
# Launches: Gazebo simulation, traffic, ML ADAS, behavior generator, evaluator
set -e
cd ~/adas_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "=== Starting Gazebo simulation ==="
ros2 launch adas_project sim.launch.py &
SIM_PID=$!
sleep 12

echo "=== Starting ADAS nodes (ML mode) ==="
ros2 run adas_project scenario_controller.py &
ros2 run adas_project traffic_behavior_node.py &
ros2 run adas_project inference_node.py &
ros2 run adas_project ml_adas.py &
ros2 run adas_project control_arbitration_node.py &
sleep 3

echo "=== Starting Autonomous Driving Agent ==="
ros2 run adas_project behavior_generator.py &
ros2 run adas_project dataset_exporter.py &
sleep 2

echo "=== Starting Evaluator ==="
echo "Press Ctrl+C to stop and generate report."
ros2 run adas_project evaluator.py

# Cleanup on exit
kill $SIM_PID 2>/dev/null
pkill -f adas_project 2>/dev/null
echo "Done."
