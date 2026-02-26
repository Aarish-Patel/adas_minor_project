#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetEntityState
import json
import csv
import os
import time

class DatasetOrchestrator(Node):
    def __init__(self):
        super().__init__('dataset_orchestrator')
        
        self.declare_parameter('episodes_per_driver', 25)
        self.declare_parameter('driver_type', 'NORMAL')
        
        self.episodes_target = self.get_parameter('episodes_per_driver').value
        self.driver_type = self.get_parameter('driver_type').value
        
        self.sub = self.create_subscription(String, '/driver_telemetry', self.telemetry_cb, 10)
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        # We will append to dataset file
        self.dataset_dir = '/home/hsiraa/adas_ws/src/dataset'
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.csv_file = os.path.join(self.dataset_dir, 'adas_dataset.csv')
        self.headers = ['speed', 'acceleration', 'lateral_deviation', 'yaw', 'steering', 
                        'throttle', 'brake', 'obstacle_distance', 'ttc', 'offroad', 'collision', 
                        'driver_type', 'intent_lane_change', 'intent_brake', 'intent_overtake', 'intent_offroad']
        
        # Write headers if file doesn't exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                
        self.current_ep = 0
        self.step_count = 0
        self.max_steps = 2400 # Max steps per episode (~120s sim time)
        self.current_episode_data = []
        
        self.get_logger().info(f"Orchestrator online. Target: {self.episodes_target} eps of '{self.driver_type}'")
        self.reset_simulation()
        
    def reset_simulation(self):
        self.get_logger().info(f"Resetting Episode {self.current_ep + 1}/{self.episodes_target}")
        
        # Robust initialization of state message
        req = SetEntityState.Request()
        req.state.name = 'adas_vehicle'
        req.state.pose.position.x = -50.0
        req.state.pose.position.y = 46.5
        req.state.pose.position.z = 0.5
        # Quaternion for yaw = 0 (facing East, CW track direction)
        req.state.pose.orientation.z = 0.0
        req.state.pose.orientation.w = 1.0
        
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        
        # Wait for service and call with block/retry
        if not self.set_state_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Reset service not available!")
            return

        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None and future.result().success:
            self.step_count = 0
            self.current_episode_data = []
            self.collision_delay = 5  # Give it a few ticks to clear old states
        else:
            self.get_logger().warning("Failed to reset simulation, will retry next tick.")
        
        # Sleep slightly to let physics settle
        time.sleep(2.0)
        
    def save_episode(self):
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            for row in self.current_episode_data:
                writer.writerow(row)
                
    def telemetry_cb(self, msg):
        if self.current_ep >= self.episodes_target:
            self.get_logger().info("Target Reached. Shutting down orchestrator.")
            raise SystemExit # Clean exit
            
        data = json.loads(msg.data)
        
        if getattr(self, 'collision_delay', 0) > 0:
            self.collision_delay -= 1
            return

        # Store in buffer
        self.current_episode_data.append(data)
        self.step_count += 1
        
        # Stop condition
        # Important: Don't trigger resets in the highly unstable first few ticks
        if self.step_count > 10 and (data['collision'] or data['offroad']):
            self.save_episode()
            self.current_ep += 1
            self.reset_simulation()
        elif self.step_count >= self.max_steps:
            self.save_episode()
            self.current_ep += 1
            self.reset_simulation()

def main(args=None):
    rclpy.init(args=args)
    node = DatasetOrchestrator()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
