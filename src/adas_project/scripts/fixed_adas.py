#!/usr/bin/env python3
"""
fixed_adas.py — Classical Fixed-Threshold ADAS System.
Implements 5 ADAS Levels strictly using predefined physical thresholds.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
import json

class FixedADAS(Node):
    def __init__(self):
        super().__init__('fixed_adas')
        
        self.declare_parameter('adas_level', 2)
        self.level = self.get_parameter('adas_level').value
        
        self.sub = self.create_subscription(String, '/driver_telemetry', self.driver_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.metrics_pub = self.create_publisher(Float32, '/adas/interruptions', 10)
        
        self.target_speed = 15.0 # Cruise control speed
        self.get_logger().info(f"Fixed ADAS Active. Level {self.level}")
        self.interruptions = 0.0

    def driver_cb(self, msg):
        state = json.loads(msg.data)
        
        speed = state['speed']
        ttc = state['ttc']
        lane = state['lateral_deviation']
        obs_dist = state['obstacle_distance']
        
        user_vel = (state['throttle'] * 3.0 - state['brake'] * 6.0) # From pseudo
        # We need the actual user target velocity that behavior gen wanted. 
        # But we can just calculate ADAS velocity commands and overwrite.
        
        steer = state['steering']
        throttle = state['throttle']
        brake = state['brake']
        
        # Determine base velocity from throttle and brake
        vel_cmd = speed + (throttle * 3.0 - brake * 6.0) * 0.1
        vel_cmd = max(0.0, vel_cmd)
        
        orig_steer = steer
        orig_vel = vel_cmd
        
        # Level 1: Speed maintain (Cruise Control)
        if self.level >= 1:
            if speed < self.target_speed and obs_dist > 30.0:
                vel_cmd = max(vel_cmd, self.target_speed)
                
        # Level 2: Lane keeping + emergency stop
        if self.level >= 2:
            if abs(lane) > 3.0:
                steer = -0.15 * lane
            
            if ttc < 2.0 or obs_dist < 8.0:
                vel_cmd = 0.0
                
        # Level 3: Obstacle avoidance 
        if self.level >= 3:
            if ttc < 3.5 and obs_dist < 25.0:
                steer = 0.4 if lane < 2.0 else -0.4
                
        # Level 4: Full autonomy
        if self.level == 4:
            steer = -0.1 * lane
            
            if obs_dist < 15.0 or ttc < 2.5:
                vel_cmd = 0.0
            else:
                vel_cmd = self.target_speed
                
            if obs_dist < 30.0 and speed > 5.0:
                steer = 0.2 if lane < 2.0 else -0.2
                
        # Hard Safety Override ALWAYS dominates for Level 2+
        if self.level >= 2:
            if ttc < 1.0 or obs_dist < 5.0:
                vel_cmd = 0.0
            if abs(lane) > 5.0:
                steer = -0.3 * lane
                
        # Log interruptions metrics
        if abs(orig_steer - steer) > 0.05 or abs(orig_vel - vel_cmd) > 1.0:
            self.interruptions += 1.0
            self.metrics_pub.publish(Float32(data=self.interruptions))

        cmd = Twist()
        cmd.linear.x = vel_cmd
        cmd.angular.z = steer
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = FixedADAS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
