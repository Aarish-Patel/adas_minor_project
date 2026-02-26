#!/usr/bin/env python3
"""
supervisor.py
Subscribes to /driver_telemetry and monitors the collision and offroad flags.
If the driver goes offroad or collides with an obstacle, it automatically resets the Gazebo world.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Empty
import json

class SupervisorNode(Node):
    def __init__(self):
        super().__init__('supervisor')
        
        self.sub = self.create_subscription(String, '/driver_telemetry', self.telemetry_cb, 10)
        
        self.reset_client = self.create_client(Empty, '/reset_world')
        
        # Wait for reset service
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /reset_world service...')
            
        self.offroad_ticks = 0
        self.collision_ticks = 0
        # Wait for telemetry to settle
        self.warmup_ticks = 20
        self.resetting = False
        
        self.get_logger().info('Supervisor active. Monitoring for collisions and offroad events.')

    def telemetry_cb(self, msg):
        if self.warmup_ticks > 0:
            self.warmup_ticks -= 1
            return
            
        if self.resetting:
            return
            
        try:
            state = json.loads(msg.data)
            offroad = state.get('offroad', 0)
            collision = state.get('collision', 0)
            
            if offroad:
                self.offroad_ticks += 1
            else:
                self.offroad_ticks = 0
                
            if collision:
                self.collision_ticks += 1
            else:
                self.collision_ticks = 0
                
            # If offroad for > 10 ticks (0.5s at 20Hz) or collision for 5 ticks, reset
            if self.offroad_ticks > 10:
                self.get_logger().warn('Driver went OFFROAD! Resetting simulation...')
                self.reset_sim()
            elif self.collision_ticks > 5:
                self.get_logger().warn('Driver COLLIDED! Resetting simulation...')
                self.reset_sim()
                
        except Exception as e:
            self.get_logger().error(f"Error parsing telemetry: {e}")

    def reset_sim(self):
        self.resetting = True
        req = Empty.Request()
        
        # Reset the world synchronously or asynchronously
        future = self.reset_client.call_async(req)
        future.add_done_callback(self.reset_done_cb)
        
    def reset_done_cb(self, future):
        try:
            future.result()
            self.get_logger().info('Simulation reset successfully.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            
        # Give simulation time to settle before monitoring again
        self.warmup_ticks = 20
        self.offroad_ticks = 0
        self.collision_ticks = 0
        self.resetting = False

def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
