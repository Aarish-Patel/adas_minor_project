#!/bin/bash
# manual_ml.sh — Manual WASD driving with ML-adaptive ADAS
# Launches: Gazebo simulation, traffic, ADAS (ML mode), manual driver
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
ros2 run adas_project fixed_adas.py &
ros2 run adas_project ml_adas.py &
ros2 run adas_project control_arbitration_node.py &
sleep 3

echo "=== Starting Manual Driver (ML ADAS) ==="
echo ">>> WASD controls in THIS terminal <<<"
echo "    W=accel  S=brake  A=left  D=right  SPACE=stop  Q=quit"
ros2 run adas_project manual_driver.py --mode ml

# Cleanup on exit
kill $SIM_PID 2>/dev/null
pkill -f adas_project 2>/dev/null
echo "Done."
