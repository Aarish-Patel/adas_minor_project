#!/bin/bash
# train.sh — Collect data then train the ML intent model
# Step 1: Run autonomous agent to collect dataset
# Step 2: Train GRU model on the collected data
set -e
cd ~/adas_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "=============================="
echo " STEP 1: Data Collection"
echo "=============================="
echo "Launching Gazebo + autonomous agent for data collection..."

ros2 launch adas_project sim.launch.py &
SIM_PID=$!
sleep 12

ros2 run adas_project scenario_controller.py &
ros2 run adas_project traffic_behavior_node.py &
ros2 run adas_project fixed_adas.py &
ros2 run adas_project behavior_generator.py &
ros2 run adas_project dataset_exporter.py &
sleep 2

echo "Collecting data... Let it run for 2-5 minutes."
echo "Press Ctrl+C when enough data is collected."
wait $! 2>/dev/null || true

# Kill simulation
kill $SIM_PID 2>/dev/null
pkill -f adas_project 2>/dev/null
pkill -9 -f gazebo 2>/dev/null
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f gzclient 2>/dev/null
sleep 3

echo ""
echo "=============================="
echo " STEP 2: Training Model"
echo "=============================="

if [ ! -f ~/adas_ws/dataset/adas_features.csv ]; then
    echo "ERROR: No dataset found at ~/adas_ws/dataset/adas_features.csv"
    echo "Run data collection first."
    exit 1
fi

echo "Training GRU intent model..."
python3 ~/adas_ws/src/adas_project/scripts/train_model.py

echo ""
echo "=============================="
echo " Training Complete!"
echo "=============================="
echo "Model saved to: intent_model.pt"
echo "You can now run auto_ml.sh or manual_ml.sh"
