#!/bin/bash
# collect_data.sh — Launches the Gazebo simulation + Fixed ADAS + Dataset Exporter
# Runs for a set duration to gather realistic physics data, then cleanly exits.

bold="\e[1m"
green="\e[32m"
red="\e[31m"
reset="\e[0m"

echo -e "${bold}${green}=== ADAS Dataset Collection ===${reset}"
echo "Sourcing ROS 2 workspace..."
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "Clearing old dataset..."
rm -f dataset/adas_features.csv

echo "Starting Gazebo simulation..."
ros2 launch adas_project sim.launch.py &
GAZEBO_PID=$!

echo "Waiting for Gazebo to initialize..."
sleep 15

# Start Fixed ADAS and Behavior Generator (Auto Mode)
ros2 run adas_project fixed_adas.py &
FIXED_PID=$!

ros2 run adas_project behavior_generator.py --mode auto &
BEHAVIOR_PID=$!

echo "Spawning dynamic traffic and scenarios..."
ros2 run adas_project scenario_controller.py &
SCENARIO_PID=$!

ros2 run adas_project traffic_behavior_node.py &
TRAFFIC_PID=$!

echo "Starting Dataset Exporter..."
ros2 run adas_project dataset_exporter.py &
EXPORTER_PID=$!

echo -e "\n${bold}>>> Dataset collection running for 3 minutes... <<<${reset}\n"

# Run for 180 seconds (3 minutes) to gather a solid dataset
sleep 180

echo -e "\n${bold}${red}Cleaning up and killing processes...${reset}"
kill $EXPORTER_PID 2>/dev/null
kill $TRAFFIC_PID 2>/dev/null
kill $SCENARIO_PID 2>/dev/null
kill $BEHAVIOR_PID 2>/dev/null
kill $FIXED_PID 2>/dev/null
kill $GAZEBO_PID 2>/dev/null

# Kill any stubborn ROS/Gazebo processes
pkill -9 -f gazebo
pkill -9 -f gzserver
pkill -9 -f gzclient
pkill -9 -f ros2

echo -e "${bold}${green}Dataset Collection Complete.${reset}"
ls -lh dataset/adas_features.csv
