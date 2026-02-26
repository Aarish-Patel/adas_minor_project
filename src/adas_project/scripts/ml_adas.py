#!/usr/bin/env python3
"""
ml_adas.py — ML-Adaptive Intent-Aware ADAS.
Dynamically modulates safety thresholds based on inferred driver intent.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
import json
import os
import pickle
import numpy as np

class MLADAS(Node):
    def __init__(self):
        super().__init__('ml_adas')
        
        self.declare_parameter('adas_level', 2)
        self.level = self.get_parameter('adas_level').value
        
        self.sub = self.create_subscription(String, '/driver_telemetry', self.driver_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.metrics_pub = self.create_publisher(Float32, '/adas/interruptions', 10)
        
        self.interruptions = 0.0
        self.target_speed = 15.0
        
        # Load models
        self.models = {}
        model_dir = os.path.join(os.path.dirname(__file__), '../models')
        targets = ['intent_lane_change', 'intent_brake', 'intent_overtake', 'intent_offroad']
        
        if os.path.exists(model_dir):
            for t in targets:
                path = os.path.join(model_dir, f'{t}_rf.pkl')
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.models[t] = pickle.load(f)
                        
        self.get_logger().info(f"ML ADAS Active. Level {self.level}. Loaded {len(self.models)} models.")

    def _predict_intents(self, state):
        if not self.models:
            return 0, 0, 0, 0
            
        # Features: [speed, acceleration, lateral_deviation, yaw, steering, throttle, brake, obstacle_distance, ttc]
        # Our training used: ['speed', 'acceleration', 'lateral_deviation', 'yaw', 'steering', 'throttle', 'brake', 'obstacle_distance', 'ttc']
        accel = state['throttle'] * 3.0 - state['brake'] * 6.0
        features = np.array([[
            state['speed'], accel, state['lateral_deviation'], state['yaw'], 
            state['steering'], state['throttle'], state['brake'], 
            state['obstacle_distance'], state['ttc'] if state['ttc'] < 900 else 999.0
        ]])
        
        try:
            p_lc = self.models['intent_lane_change'].predict(features)[0]
            p_br = self.models['intent_brake'].predict(features)[0]
            p_ov = self.models['intent_overtake'].predict(features)[0]
            p_of = self.models['intent_offroad'].predict(features)[0]
            return p_lc, p_br, p_ov, p_of
        except Exception as e:
            self.get_logger().error(f"Prediction failed: {e}")
            return 0, 0, 0, 0

    def driver_cb(self, msg):
        state = json.loads(msg.data)
        
        speed = state['speed']
        ttc = state['ttc']
        lane = state['lateral_deviation']
        obs_dist = state['obstacle_distance']
        
        steer = state['steering']
        throttle = state['throttle']
        brake = state['brake']
        
        vel_cmd = speed + (throttle * 3.0 - brake * 6.0) * 0.1
        vel_cmd = max(0.0, vel_cmd)
        
        orig_steer = steer
        orig_vel = vel_cmd
        
        # Infer Intents
        p_lc, p_br, p_ov, p_of = self._predict_intents(state)
        
        # Adaptive Thresholds
        ttc_thresh = 2.0
        lane_thresh = 3.0
        obs_thresh = 8.0
        
        if p_lc > 0.5 or p_ov > 0.5:     # Driver wants to maneuver
            lane_thresh = 5.5
        if p_br > 0.5:                   # Prepare for braking
            vel_cmd *= 0.5
        if p_of > 0.5:                   # Driver erratic -> tighten
            lane_thresh = 1.5
            
        # Level 1: Speed maintain
        if self.level >= 1:
            if speed < self.target_speed and obs_dist > 30.0:
                vel_cmd = max(vel_cmd, self.target_speed)
                
        # Level 2: Lane keeping + emergency stop
        if self.level >= 2:
            if abs(lane) > lane_thresh:
                steer = -0.15 * lane
            
            if ttc < ttc_thresh or obs_dist < obs_thresh:
                vel_cmd = 0.0
                
        # Level 3: Obstacle avoidance 
        if self.level >= 3:
            if ttc < (ttc_thresh + 1.5) and obs_dist < (obs_thresh + 15.0):
                if p_lc < 0.5 and p_ov < 0.5:
                    steer = 0.4 if lane < 2.0 else -0.4
                    
        # Level 4: Full autonomy
        if self.level == 4:
            if p_lc < 0.5:
                steer = -0.1 * lane
            
            if obs_dist < 15.0 or ttc < 2.5:
                if p_ov > 0.5:
                    steer = 0.2 if lane < 2.0 else -0.2
                    vel_cmd = speed * 0.5
                else:    
                    vel_cmd = 0.0
            else:
                vel_cmd = self.target_speed
                
        # Hard Safety Override ALWAYS dominates for Level 2+
        if self.level >= 2:
            if ttc < 1.0 or obs_dist < 5.0:
                vel_cmd = 0.0
            if abs(lane) > 5.5:
                steer = -0.3 * lane
                
        if abs(orig_steer - steer) > 0.05 or abs(orig_vel - vel_cmd) > 1.0:
            self.interruptions += 1.0
            self.metrics_pub.publish(Float32(data=self.interruptions))

        cmd = Twist()
        cmd.linear.x = float(vel_cmd)
        cmd.angular.z = float(steer)
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = MLADAS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
