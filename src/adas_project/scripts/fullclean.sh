#!/bin/bash
# fullclean.sh — Kill ALL project-related processes
echo "=== Killing all ADAS project processes ==="

pkill -9 -f gazebo 2>/dev/null
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f gzclient 2>/dev/null
pkill -9 -f adas_project 2>/dev/null
pkill -9 -f manual_driver 2>/dev/null
pkill -9 -f behavior_generator 2>/dev/null
pkill -9 -f scenario_controller 2>/dev/null
pkill -9 -f traffic_behavior 2>/dev/null
pkill -9 -f inference_node 2>/dev/null
pkill -9 -f fixed_adas 2>/dev/null
pkill -9 -f ml_adas 2>/dev/null
pkill -9 -f control_arbitration 2>/dev/null
pkill -9 -f dataset_exporter 2>/dev/null
pkill -9 -f evaluator 2>/dev/null
pkill -9 -f robot_state_publisher 2>/dev/null
pkill -9 -f ros2 2>/dev/null

sleep 2
echo "=== All processes killed ==="
echo "Run 'ros2 daemon start' if ROS2 commands stop working."
