#!/usr/bin/env python3
"""
ml_adas.py — ML-Adaptive ADAS with Real-time Explainability.

This node enhances classical ADAS by integrating an intent prediction GRU model.
It dynamically adjusts safety thresholds based on the predicted intent of 
surrounding actors or the ego-vehicle's own risky behaviors.

Key Features:
- Dynamic TTC Adjustment: Shrinks the safety envelope during 'Aggressive' intent.
- Explainable AI (XAI): Publishes intent probabilities and RViz text markers.
- Hybrid Logic: Combines classical lane-keeping with ML-driven sensitivity.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker
import math

# Track and Model Config
TRACK_CENTER = 48.5
CLASS_NAMES  = ['Aggressive', 'LaneChange', 'Brake', 'Normal']

# Static Safety Fallbacks
TTC_WARNING   = 2.5
TTC_ASSIST    = 1.5
TTC_EMERGENCY = 0.8
LANE_THRESHOLD = 3.0
STEERING_ACTIVE_THRESHOLD = 0.05


class MLADAS(Node):
    """
    ROS2 Node for Machine Learning driven ADAS.
    
    Dynamically modulates collision intervention sensitivity using 
    deep learning intent probabilities.
    """
    def __init__(self):
        super().__init__('ml_adas')

        # ─── Initialization ──────────────────────────────────────────
        # Sensor and Model Subscriptions
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)
        self.create_subscription(Float32, '/intent_probability', self.intent_scalar_cb, 10)
        self.create_subscription(Float32MultiArray, '/intent_probabilities', self.intent_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        # ADAS Command Publishers
        self.alert_pub  = self.create_publisher(Bool, '/adas/alert/ml', 10)
        self.ttc_pub    = self.create_publisher(Float32, '/adas/ttc/ml', 10)
        self.reason_pub = self.create_publisher(String, '/adas/reason/ml', 10)
        self.state_pub  = self.create_publisher(String, '/adas/state/ml', 10)

        # Explainability Data Publishers
        self.prediction_pub = self.create_publisher(String, '/intent_prediction', 10)
        self.marker_pub     = self.create_publisher(Marker, '/adas/intent_marker', 10)

        # ─── Internal State ──────────────────────────────────────────
        self.velocity     = 0.0
        self.min_distance = 100.0
        self.lane_dev     = 0.0
        self.steering_rate = 0.0
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.last_steer_pos = 0.0
        
        # ML Logic Parameters
        self.intent_prob  = 0.0                  # Scalar probability of 'Aggressive' intent
        self.class_probs  = [0.0, 0.0, 0.0, 1.0] # [Aggro, LaneChange, Brake, Normal]
        self.alpha        = 0.5                  # Sensitivity scaling factor (0.0 to 1.0)
        self.base_ttc     = TTC_WARNING          # The anchor for dynamic thresholding

        self.timer = self.create_timer(0.1, self.check_alerts)
        self.log_counter = 0
        self.get_logger().info('ML-ADAS active — Dynamic TTC + XAI Interface')

    def scan_cb(self, msg):
        """Processes LiDAR data for obstacle proximity."""
        mid = len(msg.ranges) // 2
        vals = [r for r in msg.ranges[mid-20:mid+20]
                if not math.isinf(r) and not math.isnan(r) and r > 0.1]
        self.min_distance = min(vals) if vals else 100.0

    def joint_cb(self, msg):
        """Tracks steering rate for lane-change intent gating."""
        if 'front_left_steer_joint' in msg.name:
            idx = msg.name.index('front_left_steer_joint')
            pos = msg.position[idx]
            self.steering_rate = abs(pos - self.last_steer_pos) / 0.1
            self.last_steer_pos = pos

    def model_cb(self, msg):
        """Extracts ego state and calculates geometric lane deviation."""
        if 'adas_vehicle' in msg.name:
            idx = msg.name.index('adas_vehicle')
            vx = msg.twist[idx].linear.x
            vy = msg.twist[idx].linear.y
            self.velocity = math.sqrt(vx*vx + vy*vy)
            self.ego_x = msg.pose[idx].position.x
            self.ego_y = msg.pose[idx].position.y

            cx, cy = self.ego_x, self.ego_y
            if cy > 0 and -100 <= cx <= 100:
                self.lane_dev = abs(cy - TRACK_CENTER)
            elif cy <= 0 and -100 <= cx <= 100:
                self.lane_dev = abs(cy + TRACK_CENTER)
            elif cx > 100:
                self.lane_dev = abs(math.sqrt((cx-100)**2 + cy**2) - TRACK_CENTER)
            else:
                self.lane_dev = abs(math.sqrt((cx+100)**2 + cy**2) - TRACK_CENTER)

    def intent_scalar_cb(self, msg):
        """Callback for high-level Aggressive Intent probability."""
        self.intent_prob = msg.data

    def intent_cb(self, msg):
        """Callback for full categorical intent weight distribution."""
        if len(msg.data) >= 4:
            self.class_probs = list(msg.data[:4])

    def check_alerts(self):
        """Core ML-ADAS Arbitration: Modulates safety based on intent."""
        alert = False
        reason = ''
        adas_state = 'MANUAL_ONLY'

        # ─── Dynamic Thresholding Algorithm ──────────────────────────
        # Threshold moves based on intent: High Intent -> More Sensitive
        # Logic: dyn_ttc = base_ttc * (1.0 - alpha * intent_prob)
        dynamic_ttc = self.base_ttc * (1.0 - self.alpha * self.intent_prob)
        dynamic_ttc = max(dynamic_ttc, TTC_EMERGENCY) # Cap at emergency floor

        ttc = 999.0
        if self.velocity > 0.5 and self.min_distance < 50.0:
            ttc = self.min_distance / self.velocity

        # Evaluate against the DYNAMICALLY SHIFTED thresholds
        if ttc < TTC_EMERGENCY:
            alert = True
            adas_state = 'EMERGENCY_BRAKE'
            reason = f'TTC={ttc:.1f}s EMERGENCY (Dyn. Floor reached)'
        elif ttc < min(TTC_ASSIST, dynamic_ttc * 0.6):
            alert = True
            adas_state = 'ASSIST'
            reason = f'TTC={ttc:.1f}s ASSIST (ML Adjusted)'
        elif ttc < dynamic_ttc:
            alert = True
            adas_state = 'WARNING'
            reason = f'TTC={ttc:.1f}s WARNING (Intent-Preemptive)'

        # ─── Intent-Aware Lane Deviation ─────────────────────────────
        # Widen the lane threshold if aggressive intent is verified 
        # (allows for tighter maneuvering without nuisance alarms)
        adj_lane_thresh = LANE_THRESHOLD * (1.0 + 0.3 * self.intent_prob)
        if self.lane_dev > adj_lane_thresh:
            if self.steering_rate < STEERING_ACTIVE_THRESHOLD:
                alert = True
                reason += f' | Unintentional Drift: {self.lane_dev:.1f}m'

        # ─── Publishing results ───────────────────────────────────────
        self.alert_pub.publish(Bool(data=alert))
        self.ttc_pub.publish(Float32(data=float(ttc)))
        self.state_pub.publish(String(data=adas_state))
        if reason:
            self.reason_pub.publish(String(data=reason))

        # Update XAI interfaces (RViz + Terminal)
        self._publish_explainability()

    def _publish_explainability(self):
        """Publishes intent data for human oversight and debugging."""
        # 1. Publish categorical prediction string
        parts = [f'{CLASS_NAMES[i]}: {self.class_probs[i]:.2f}'
                 for i in range(len(CLASS_NAMES))]
        pred_str = ' | '.join(parts)
        self.prediction_pub.publish(String(data=pred_str))

        # 2. Log to console at a throttled rate (every 2.0s)
        self.log_counter += 1
        if self.log_counter % 20 == 0:
            self.get_logger().info(f'[ML-INSIGHT] {pred_str}')

        # 3. VR/RViz 3D Text Marker above the vehicle
        best_idx = self.class_probs.index(max(self.class_probs))
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'intent'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 3.0    # 3 meters above the roof
        marker.scale.z = 0.8            # Font size
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.text = f'INTENT: {CLASS_NAMES[best_idx]} ({self.class_probs[best_idx]:.2f})'
        marker.lifetime.sec = 1
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = MLADAS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
