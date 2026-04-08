#!/usr/bin/env python3
"""
cmd_vel_relay.py — Relays /cmd_vel to /ackermann_steering_controller/cmd_vel_unstamped

The Nav2 stack publishes velocity commands on /cmd_vel but the
ros2_control ackermann_steering_controller subscribes on its own
namespaced topic. This relay bridges the two.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelRelay(Node):
    def __init__(self):
        super().__init__('cmd_vel_relay')
        self._pub = self.create_publisher(
            Twist, '/ackermann_steering_controller/cmd_vel_unstamped', 10)
        self._sub = self.create_subscription(
            Twist, '/cmd_vel', self._cb, 10)
        self.get_logger().info(
            'Relaying /cmd_vel → /ackermann_steering_controller/cmd_vel_unstamped')

    def _cb(self, msg: Twist):
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
