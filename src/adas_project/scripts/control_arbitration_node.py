#!/usr/bin/env python3
"""
control_arbitration_node.py — Safety Arbitration FSM.

This node acts as the centralized safety supervisor. It monitors the unified 
TTC (Time-to-Collision) and manages the transition between manual driving 
and active safety overrides.

FSM States:
- MANUAL_ONLY: Driver has full control.
- WARNING: Passive visual/haptic alerts provided.
- ASSIST: Active steering/throttle correction enabled.
- EMERGENCY_BRAKE: Full safety override (Force stop).
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import math
import time

# State Machine Constants
TTC_WARNING   = 4.0
TTC_ASSIST    = 2.5
TTC_EMERGENCY = 1.3
HYSTERESIS_T  = 0.3    # Minimum time to stay in a state to avoid rapid flapping

STATES = ['MANUAL_ONLY', 'WARNING', 'ASSIST', 'EMERGENCY_BRAKE']


class ControlArbitrationNode(Node):
    """
    ROS2 Node for Safety Arbitration.
    
    Implements a robust Hysteresis-aware Finite State Machine to manage 
    ADAS intervention levels.
    """
    def __init__(self):
        super().__init__('control_arbitration')

        # ─── Initialization ──────────────────────────────────────────
        # Sensor Subscriptions
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)
        
        # Monitor Both Classical and ML ADAS outputs
        self.create_subscription(Bool, '/adas/alert/fixed', self.alert_fixed_cb, 10)
        self.create_subscription(Bool, '/adas/alert/ml', self.alert_ml_cb, 10)

        # Output Command Channels
        self.cmd_pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/adas/state', 10)
        self.ttc_pub   = self.create_publisher(Float32, '/adas/ttc', 10)

        # ─── Internal State ──────────────────────────────────────────
        self.state = 'MANUAL_ONLY'
        self.prev_state = 'MANUAL_ONLY'
        self.state_enter_time = time.time()
        
        self.velocity     = 0.0
        self.min_distance = 100.0
        self.ttc          = 999.0
        
        self.alert_fixed  = False
        self.alert_ml     = False

        # FSM Update Loop (20Hz for low latency intervention)
        self.timer = self.create_timer(0.05, self.arbitrate)
        self.get_logger().info(
            f'Safety FSM Active — Mode: Hysteresis-Aware Arbitration')

    def scan_cb(self, msg):
        """Processes LiDAR data for unified TTC calculation."""
        mid = len(msg.ranges) // 2
        vals = [r for r in msg.ranges[mid-20:mid+20]
                if not math.isinf(r) and not math.isnan(r) and r > 0.1]
        self.min_distance = min(vals) if vals else 100.0

    def model_cb(self, msg):
        """Extracts ego ground-truth velocity."""
        if 'adas_vehicle' in msg.name:
            idx = msg.name.index('adas_vehicle')
            vx = msg.twist[idx].linear.x
            vy = msg.twist[idx].linear.y
            self.velocity = math.sqrt(vx*vx + vy*vy)

    def alert_fixed_cb(self, msg):
        self.alert_fixed = msg.data

    def alert_ml_cb(self, msg):
        self.alert_ml = msg.data

    def arbitrate(self):
        """Main FSM Transition Logic Loop."""
        now = time.time()

        # 1. Compute Ground Truth TTC
        if self.velocity > 0.5 and self.min_distance < 50.0:
            self.ttc = self.min_distance / self.velocity
        else:
            self.ttc = 999.0

        self.ttc_pub.publish(Float32(data=float(self.ttc)))

        # 2. Determine Target State based on physical safety metrics
        target_state = 'MANUAL_ONLY'
        if self.ttc < TTC_EMERGENCY:
            target_state = 'EMERGENCY_BRAKE'
        elif self.ttc < TTC_ASSIST:
            target_state = 'ASSIST'
        elif self.ttc < TTC_WARNING:
            target_state = 'WARNING'

        # 3. Apply Hysteresis and State Transitions
        # We always allow instantaneous escalation to higher safety tiers.
        # We enforce a delay (HYSTERESIS_T) before de-escalating to manual.
        state_idx_current = STATES.index(self.state)
        state_idx_target  = STATES.index(target_state)

        if state_idx_target > state_idx_current:
            # Escalation: Instant
            self.state = target_state
            self.state_enter_time = now
        elif state_idx_target < state_idx_current:
            # De-escalation: Wait for cooldown
            if (now - self.state_enter_time) > HYSTERESIS_T:
                self.state = target_state
                self.state_enter_time = now

        # 4. Critical Safety Actions
        if self.state == 'EMERGENCY_BRAKE':
            # Override all inputs: Set velocity to DEAD STOP
            stop = Twist()
            self.cmd_pub.publish(stop)

        # 5. Telemetry and State Broadcast
        if self.state != self.prev_state:
            self.get_logger().info(f'[FSM TRANSITION] {self.prev_state} ➔ {self.state} (TTC: {self.ttc:.2f}s)')
            self.prev_state = self.state

        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControlArbitrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
