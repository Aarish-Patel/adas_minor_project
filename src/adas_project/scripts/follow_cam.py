#!/usr/bin/env python3
"""
follow_cam.py — Sets the Gazebo GUI camera to follow the ego vehicle
from a nice 3rd-person perspective above and behind.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import subprocess, math

class FollowCam(Node):
    def __init__(self):
        super().__init__('follow_cam')
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_timer(0.2, self.update_cam)  # 5 Hz camera updates
        self.x, self.y, self.yaw = -50.0, 46.5, 0.0

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2))

    def update_cam(self):
        # Camera 15m behind, 10m above, looking slightly down
        cx = self.x - 15.0 * math.cos(self.yaw)
        cy = self.y - 15.0 * math.sin(self.yaw)
        cz = 10.0
        # Look-at angles: pitch down ~30°, yaw matches vehicle
        pitch = 0.5  # ~30 degrees down
        try:
            subprocess.run([
                'gz', 'camera', '-c', 'gzclient_camera',
                '-s', f'{cx} {cy} {cz} 0 {pitch} {self.yaw}'
            ], timeout=0.5, capture_output=True)
        except Exception:
            pass

def main():
    rclpy.init()
    node = FollowCam()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
