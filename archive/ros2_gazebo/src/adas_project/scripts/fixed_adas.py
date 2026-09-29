#!/usr/bin/env python3
"""
fixed_adas.py — Classical Fixed-Threshold ADAS System.

This node implements a rule-based Advanced Driver Assistance System (ADAS)
using Time-to-Collision (TTC) and Lane Deviation metrics. It provides 
multi-tier alerts (Warning, Assist, Emergency Brake) based on static 
safety thresholds.

Key Features:
- TTC-based collision avoidance using LiDAR data.
- Lane Keeping Assistance (LKA) with intentionality detection 
  (steering-rate gating).
- ROS2 humble integration with standard sensor and robot messages.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, JointState
from std_msgs.msg import Bool, Float32, String
from gazebo_msgs.msg import ModelStates
import math

# Track Geometry: Central Y-coordinate of the track loops
TRACK_CENTER = 48.5

# Safety Thresholds (Seconds)
TTC_WARNING   = 2.5    # Trigger visual/audible alert
TTC_ASSIST    = 1.5    # Trigger active assistance/braking
TTC_EMERGENCY = 0.8    # Trigger full emergency braking

# Lane Deviation Threshold (Meters)
LANE_THRESHOLD = 3.0

# Steering intentionality threshold (rad/s)
# Above this rate, we assume the driver is intentionally changing lanes.
STEERING_ACTIVE_THRESHOLD = 0.05


class FixedADAS(Node):
    """
    ROS2 Node for classical ADAS logic.
    
    Subscribes to sensor data and publishes ADAS states and alerts based 
    on fixed physical thresholds.
    """
    def __init__(self):
        super().__init__('fixed_adas')

        # ─── Initialization ──────────────────────────────────────────
        # Subscriptions: LiDAR, Ground Truth (Gazebo), and Robot Internals
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        # Publishers: Alerts, Raw TTC, Reason for alert, and FSM State
        self.alert_pub  = self.create_publisher(Bool, '/adas/alert/fixed', 10)
        self.ttc_pub    = self.create_publisher(Float32, '/adas/ttc/fixed', 10)
        self.reason_pub = self.create_publisher(String, '/adas/reason/fixed', 10)
        self.state_pub  = self.create_publisher(String, '/adas/state/fixed', 10)

        # ─── Internal State ──────────────────────────────────────────
        self.velocity     = 0.0      # Current vehicle speed (m/s)
        self.min_distance = 100.0    # Distance to nearest obstacle ahead (m)
        self.lane_dev     = 0.0      # Current lane deviation from centerline (m)
        self.steering_rate = 0.0     # Temporal derivative of steering position (rad/s)
        self.ego_x = 0.0             # Current X position (Track coordinates)
        self.ego_y = 0.0             # Current Y position (Track coordinates)
        self.last_steer_pos = 0.0    # Previous steering angle for rate calculation

        # Main logic timer running at 10Hz
        self.timer = self.create_timer(0.1, self.check_alerts)
        self.get_logger().info(
            f'FixedADAS active — TTC thresholds: warn={TTC_WARNING}s '
            f'assist={TTC_ASSIST}s emerg={TTC_EMERGENCY}s')

    def scan_cb(self, msg):
        """Processes LiDAR data to find the nearest object in the front arc."""
        mid = len(msg.ranges) // 2
        # Analyze a 40-sample window centered at 0 degrees
        vals = [r for r in msg.ranges[mid-20:mid+20]
                if not math.isinf(r) and not math.isnan(r) and r > 0.1]
        self.min_distance = min(vals) if vals else 100.0

    def joint_cb(self, msg):
        """Calculates steering rate to determine driver intentionality."""
        if 'front_left_steer_joint' in msg.name:
            idx = msg.name.index('front_left_steer_joint')
            pos = msg.position[idx]
            # Simple numerical differentiation for steering rate
            self.steering_rate = abs(pos - self.last_steer_pos) / 0.1
            self.last_steer_pos = pos

    def model_cb(self, msg):
        """Computes vehicle velocity and track-relative lane deviation."""
        if 'adas_vehicle' in msg.name:
            idx = msg.name.index('adas_vehicle')
            vx = msg.twist[idx].linear.x
            vy = msg.twist[idx].linear.y
            self.velocity = math.sqrt(vx*vx + vy*vy)
            self.ego_x = msg.pose[idx].position.x
            self.ego_y = msg.pose[idx].position.y

            # Calculate Lane Deviation based on track geometry (2 straightaways + 2 curves)
            cx, cy = self.ego_x, self.ego_y
            if cy > 0 and -100 <= cx <= 100:      # Top Straight
                self.lane_dev = abs(cy - TRACK_CENTER)
            elif cy <= 0 and -100 <= cx <= 100:    # Bottom Straight
                self.lane_dev = abs(cy + TRACK_CENTER)
            elif cx > 100:                        # Right Curve
                self.lane_dev = abs(math.sqrt((cx-100)**2 + cy**2) - TRACK_CENTER)
            else:                                 # Left Curve
                self.lane_dev = abs(math.sqrt((cx+100)**2 + cy**2) - TRACK_CENTER)

    def check_alerts(self):
        """Main ADAS logic loop: Evaluates safety metrics and publishes alerts."""
        alert = False
        reason = ''
        adas_state = 'MANUAL_ONLY'

        # ─── Tier 1: Time-to-Collision (TTC) Logic ────────────────────
        ttc = 999.0
        if self.velocity > 0.5 and self.min_distance < 50.0:
            ttc = self.min_distance / self.velocity

        if ttc < TTC_EMERGENCY:
            alert = True
            adas_state = 'EMERGENCY_BRAKE'
            reason = f'TTC={ttc:.1f}s < {TTC_EMERGENCY}s (EMERGENCY)'
        elif ttc < TTC_ASSIST:
            alert = True
            adas_state = 'ASSIST'
            reason = f'TTC={ttc:.1f}s < {TTC_ASSIST}s (ASSIST)'
        elif ttc < TTC_WARNING:
            alert = True
            adas_state = 'WARNING'
            reason = f'TTC={ttc:.1f}s < {TTC_WARNING}s (WARNING)'

        # ─── Tier 2: Lane Keeping Logic ──────────────────────────────
        # Logic: If Deviation is high AND Steering is inactive, it's an unintentional drift.
        if self.lane_dev > LANE_THRESHOLD:
            if self.steering_rate < STEERING_ACTIVE_THRESHOLD:
                alert = True
                reason += f' | Unintentional Lane Deviation: {self.lane_dev:.1f}m'
            # Note: If steering is active, the agent assumes the driver is performing 
            # an intentional maneuver (like avoiding an obstacle) and suppresses the alarm.

        # ─── Publishing results ───────────────────────────────────────
        self.alert_pub.publish(Bool(data=alert))
        self.ttc_pub.publish(Float32(data=float(ttc)))
        self.state_pub.publish(String(data=adas_state))
        if reason:
            self.reason_pub.publish(String(data=reason))


def main(args=None):
    rclpy.init(args=args)
    node = FixedADAS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
