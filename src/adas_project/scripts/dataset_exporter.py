#!/usr/bin/env python3
"""
dataset_exporter.py — collects sensor data + active policy label and writes
a sliding-window feature CSV for offline ML training.

Features per time-step (7):
  yaw_rate, accel_x, steer_rate, vel_deriv, lane_dev, effort, min_dist

Window size: 10  →  70 feature columns + 1 label column per row.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
import csv, collections, time, os, math

TRACK_CENTER = 48.5   # y-coordinate of road centerline on top straight
NUM_FEATURES = 7
WINDOW_SIZE  = 10


class DatasetExporter(Node):
    def __init__(self):
        super().__init__('dataset_exporter')

        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(String, '/active_policy', self.policy_cb, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        self.current_policy = 'normal'
        self.yaw_rate    = 0.0
        self.accel_x     = 0.0
        self.steer_pos   = 0.0
        self.steer_rate  = 0.0
        self.joint_effort = 0.0
        self.velocity    = 0.0
        self.ego_x       = 0.0
        self.ego_y       = 0.0
        self.lane_deviation = 0.0
        self.min_distance = 100.0

        self.last_velocity = 0.0
        self.last_time = time.time()
        self.last_msg_time = time.time()

        self.feature_window = collections.deque(maxlen=WINDOW_SIZE)

        os.makedirs('dataset', exist_ok=True)
        self.csv_path = 'dataset/adas_features.csv'
        file_exists = os.path.exists(self.csv_path) and os.stat(self.csv_path).st_size > 0
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        if not file_exists:
            header = []
            for i in range(WINDOW_SIZE):
                header.extend([
                    f'yaw_rate_{i}', f'accel_{i}', f'steer_rate_{i}',
                    f'vel_deriv_{i}', f'lane_dev_{i}', f'effort_{i}', f'min_dist_{i}'
                ])
            header.append('label')
            self.csv_writer.writerow(header)

        self.timer = self.create_timer(0.1, self.extract_features)
        self.rows_written = 0
        self.get_logger().info('DatasetExporter ready — writing to dataset/adas_features.csv')

    # ── callbacks ─────────────────────────────────────────────────────

    def imu_cb(self, msg):
        self.yaw_rate = msg.angular_velocity.z
        self.accel_x  = msg.linear_acceleration.x

    def joint_cb(self, msg):
        if 'front_left_steer_joint' in msg.name:
            idx = msg.name.index('front_left_steer_joint')
            old = self.steer_pos
            self.steer_pos = msg.position[idx]
            self.steer_rate = (self.steer_pos - old) / 0.1
        if 'rear_left_wheel_joint' in msg.name:
            idx = msg.name.index('rear_left_wheel_joint')
            if len(msg.effort) > idx:
                self.joint_effort = msg.effort[idx]

    def policy_cb(self, msg):
        self.current_policy = msg.data

    def scan_cb(self, msg):
        mid = len(msg.ranges) // 2
        vals = [r for r in msg.ranges[mid-15:mid+15]
                if 0.1 < r < 200.0 and not math.isnan(r)]
        self.min_distance = min(vals) if vals else 100.0

    def model_cb(self, msg):
        self.last_msg_time = time.time()
        if 'adas_vehicle' in msg.name:
            idx = msg.name.index('adas_vehicle')
            self.ego_x = msg.pose[idx].position.x
            self.ego_y = msg.pose[idx].position.y
            vx = msg.twist[idx].linear.x
            vy = msg.twist[idx].linear.y
            self.velocity = math.sqrt(vx*vx + vy*vy)

            # Track-aware lane deviation (same geometry as behavior_generator)
            cx, cy = self.ego_x, self.ego_y
            if cy > 0 and -100 <= cx <= 100:
                self.lane_deviation = cy - TRACK_CENTER
            elif cy <= 0 and -100 <= cx <= 100:
                self.lane_deviation = -(cy + TRACK_CENTER)
            elif cx > 100:
                dist = math.sqrt((cx - 100)**2 + cy**2)
                self.lane_deviation = dist - TRACK_CENTER
            else:
                dist = math.sqrt((cx + 100)**2 + cy**2)
                self.lane_deviation = dist - TRACK_CENTER

    # ── feature extraction ────────────────────────────────────────────

    def extract_features(self):
        now = time.time()
        # If Gazebo is shut down, we stop receiving model states.
        # Don't keep appending duplicate stale features if more than 1s passes without an update.
        if now - self.last_msg_time > 1.0:
            return
            
        dt = now - self.last_time
        if dt <= 0.01:
            return
        vel_deriv = (self.velocity - self.last_velocity) / dt
        self.last_velocity = self.velocity
        self.last_time = now

        self.feature_window.append([
            self.yaw_rate,
            self.accel_x,
            self.steer_rate,
            vel_deriv,
            self.lane_deviation,
            self.joint_effort,
            self.min_distance,
        ])

        if len(self.feature_window) == WINDOW_SIZE:
            row = []
            for f in self.feature_window:
                row.extend(f)
            row.append(self.current_policy)
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            self.rows_written += 1
            if self.rows_written % 100 == 0:
                self.get_logger().info(f'[EXPORT] {self.rows_written} rows written')


def main(args=None):
    rclpy.init(args=args)
    node = DatasetExporter()
    rclpy.spin(node)
    node.csv_file.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
