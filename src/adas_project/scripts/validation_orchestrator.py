#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from gazebo_msgs.srv import SetEntityState
import json
import csv
import os
import time

class ValidationOrchestrator(Node):
    def __init__(self):
        super().__init__('validation_orchestrator')
        
        self.declare_parameter('driver_type', 'NORMAL')
        self.declare_parameter('level', 0)
        self.declare_parameter('system', 'FIXED')
        
        self.driver_type = self.get_parameter('driver_type').value
        self.level = self.get_parameter('level').value
        self.system = self.get_parameter('system').value
        
        self.episodes_target = 200 # Per combination
        
        self.sub = self.create_subscription(String, '/driver_telemetry', self.telemetry_cb, 10)
        self.int_sub = self.create_subscription(Float32, '/adas/interruptions', self.int_cb, 10)
        
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        os.makedirs('../results', exist_ok=True)
        self.csv_file = '../results/metrics.csv'
        self.headers = ['Driver', 'Level', 'System', 'Interruptions', 'AvgSpeed', 'LapTime', 
                        'ThrottleVar', 'Jerk', 'LaneVar', 'MinTTC', 'Collision', 'Offroad']
        
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                
        self.current_ep = 0
        self.step_count = 0
        self.max_steps = 300 # Shorter episodes for faster validation
        self.interruptions = 0.0
        
        # Episode Buffers
        self.ep_speed = []
        self.ep_throttle = []
        self.ep_lane = []
        self.ep_ttc = []
        
        self.get_logger().info(f"Validation Orchestrator: {self.driver_type}, L{self.level}, {self.system}")
        self.reset_simulation()
        
    def int_cb(self, msg):
        self.interruptions = msg.data

    def reset_simulation(self):
        req = SetEntityState.Request()
        req.state.name = 'adas_vehicle'
        req.state.pose.position.x = -50.0
        req.state.pose.position.y = 46.5
        req.state.pose.position.z = 0.5
        req.state.pose.orientation.z = 1.0
        req.state.pose.orientation.w = 0.0
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.angular.z = 0.0
        
        future = self.set_state_client.call_async(req)
        
        self.step_count = 0
        self.ep_speed = []
        self.ep_throttle = []
        self.ep_lane = []
        self.ep_ttc = []
        self.interruptions = 0.0
        
        time.sleep(0.5)
        
    def save_episode(self, state):
        import numpy as np
        
        avg_speed = np.mean(self.ep_speed) if self.ep_speed else 0.0
        lap_time = self.step_count * 0.1
        throttle_var = np.var(self.ep_throttle) if self.ep_throttle else 0.0
        jerk = np.var(np.diff(self.ep_speed)) if len(self.ep_speed)>1 else 0.0
        lane_var = np.var(self.ep_lane) if self.ep_lane else 0.0
        min_ttc = np.min(self.ep_ttc) if self.ep_ttc else 10.0
        
        # Write to metrics CSV
        row = {
            'Driver': self.driver_type,
            'Level': self.level,
            'System': self.system,
            'Interruptions': self.interruptions,
            'AvgSpeed': avg_speed,
            'LapTime': lap_time,
            'ThrottleVar': throttle_var,
            'Jerk': jerk,
            'LaneVar': lane_var,
            'MinTTC': min_ttc,
            'Collision': int(state['collision']),
            'Offroad': int(state['offroad'])
        }
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row)
            
        self.update_live_stats(state['collision'], state['offroad'], min_ttc)
                
    def update_live_stats(self, c, o, t):
        # Update live dashboard JSON
        try:
            path = '../results/live_stats.json'
            if os.path.exists(path):
                with open(path, 'r') as f:
                    stats = json.load(f)
            else:
                stats = {"collisions": 0, "offroads": 0, "total": 0, "current_phase": ""}
                
            stats['driver'] = self.driver_type
            stats['level'] = self.level
            stats['system'] = self.system
            stats['current_episode'] = self.current_ep
            
            if self.level >= 2:
                stats['collisions'] += int(c)
                stats['offroads'] += int(o)
                
            stats['total'] += 1
            stats['last_min_ttc'] = t
            
            with open(path, 'w') as f:
                json.dump(stats, f)
        except Exception as e:
            pass

    def telemetry_cb(self, msg):
        if self.current_ep >= self.episodes_target:
            self.get_logger().info("Combination Loop Complete. Orchestrator halting.")
            raise SystemExit
            
        data = json.loads(msg.data)
        
        self.step_count += 1
        self.ep_speed.append(data['speed'])
        self.ep_throttle.append(data['throttle'])
        self.ep_lane.append(data['lateral_deviation'])
        self.ep_ttc.append(data['ttc'] if data['ttc'] < 900 else 10.0)
        
        if self.step_count > 20 and (data['collision'] or data['offroad']):
            self.save_episode(data)
            self.current_ep += 1
            self.reset_simulation()
        elif self.step_count >= self.max_steps:
            self.save_episode(data)
            self.current_ep += 1
            self.reset_simulation()

def main(args=None):
    rclpy.init(args=args)
    node = ValidationOrchestrator()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
