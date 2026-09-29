#!/usr/bin/env python3
"""
manual_driver.py — Interactive Ego-Vehicle Teleoperation & HUD.

This node provides a human-in-the-loop interface for testing ADAS interventions.
It renders a High-Definition HUD (Heads-Up Display) over the vehicle's 
camera feed and processes real-time keyboard inputs for motion control.

Key Features:
- Keyboard Teleop: WASD for throttle/steering with momentum simulation.
- Visual HUD: Real-time telemetry (Speed, TTC, Distance, ADAS State).
- Intervention Overlays: Visual and tactical feedback during ADAS state transitions.
- Multi-Input Mixing: Blends user intent with autonomous steering assist.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
import numpy as np
import cv2
from cv_bridge import CvBridge
import math
import sys


class ManualDriver(Node):
    """
    ROS2 Node for manual control and visual telemetry feedback.
    """
    def __init__(self, adas_mode='fixed'):
        super().__init__('manual_driver')
        self.adas_mode = adas_mode

        # ─── Publishers ───
        # geometry_msgs/Twist to the standard Gazebo velocity controller
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ─── Subscribers ───
        self.create_subscription(Image, '/camera_sensor/image_raw', self.image_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)
        self.create_subscription(String, '/adas/state', self.state_cb, 10)
        self.create_subscription(Bool, f'/adas/alert/{adas_mode}', self.alert_cb, 10)
        self.create_subscription(Float32, f'/adas/ttc/{adas_mode}', self.ttc_cb, 10)
        
        # Subscribes to the ADAS suggested command for steering-assist mixing
        self.adas_cmd_sub = self.create_subscription(
            Twist, '/adas/cmd_vel', self.adas_cmd_cb, 10)

        # ─── Internal State ───
        self.bridge = CvBridge()
        self.latest_image = None
        self.linear_vel = 0.0     # Forward/Backward
        self.angular_vel = 0.0    # Steering Rate
        self.speed = 0.0          # Ground Truth
        self.min_distance = 100.0
        self.adas_alert = False
        self.ttc_value = 999.0
        self.adas_state = 'MANUAL_ONLY'
        self.active_key = ''
        self.adas_steer = 0.0

        # Orchestration: 20Hz UI and Control loop
        self.timer = self.create_timer(0.05, self.tick)
        
        self.get_logger().info(
            f'HUD initialized in {adas_mode.upper()} mode. Use WASD to navigate.')

    # ── Callbacks ─────────────────────────────────────────────────────

    def adas_cmd_cb(self, msg):
        """Captures autonomous steering intentions from the ADAS controller."""
        self.adas_steer = msg.angular.z

    def image_cb(self, msg):
        """Converts ROS Image messages to OpenCV format for HUD rendering."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except: pass

    def scan_cb(self, msg):
        """Processes LiDAR data to display front-sector obstacle distance."""
        mid = len(msg.ranges) // 2
        # Center-arc check (30-degree field of view)
        vals = [r for r in msg.ranges[mid-15:mid+15] if 0.1 < r < 200.0 and not math.isnan(r)]
        self.min_distance = min(vals) if vals else 100.0

    def model_cb(self, msg):
        """Derives absolute vehicle speed from Gazebo ground truth."""
        if 'adas_vehicle' in msg.name:
            idx = msg.name.index('adas_vehicle')
            vx = msg.twist[idx].linear.x
            vy = msg.twist[idx].linear.y
            self.speed = math.sqrt(vx*vx + vy*vy)

    def alert_cb(self, msg): self.adas_alert = msg.data
    def ttc_cb(self, msg):   self.ttc_value = msg.data
    def state_cb(self, msg): self.adas_state = msg.data

    # ── Master Tick Loop (UI + Control Arbitration) ─────────────────────

    def tick(self):
        """Main Loop: Renders the GUI and arbitrates control signals."""
        frame = self._build_frame()
        cv2.imshow('ADAS Integrated HUD', frame)
        key = cv2.waitKey(30) & 0xFF

        # Input Parsing
        if   key == ord('w'): self.active_key = 'W'
        elif key == ord('s'): self.active_key = 'S'
        elif key == ord('a'): self.active_key = 'A'
        elif key == ord('d'): self.active_key = 'D'
        elif key == ord(' '): self.active_key = 'STOP'
        elif key == ord('q'): 
            rclpy.shutdown(); return
        else: self.active_key = ''

        # ─── User Command Generation ───
        if self.active_key == 'STOP':
            self.linear_vel *= 0.3 # Friction
        elif self.active_key == 'W':
            self.linear_vel = min(self.linear_vel + 0.5, 15.0)
        elif self.active_key == 'S':
            self.linear_vel = max(self.linear_vel - 1.0, -3.0)
        else:
            self.linear_vel *= 0.995 # Air resistance

        if self.active_key == 'A':
            self.angular_vel = min(self.angular_vel + 0.1, 1.5)
        elif self.active_key == 'D':
            self.angular_vel = max(self.angular_vel - 0.1, -1.5)
        else:
            self.angular_vel *= 0.85 # Self-centering spring behavior

        # ─── ADAS Arbitration Logic ───
        # Note: This node applies the "Final Override" to the user command
        # based on the arbitration node's FSM state.
        adas_note = ""
        
        if self.adas_state == 'EMERGENCY_BRAKE':
            # Critical Safety State: Zero throttle, autonomous steering override
            self.linear_vel = 0.0
            self.angular_vel = self.adas_steer
            adas_note = ">>> ADAS OVERRIDE: BRAKING! <<<"
        
        elif self.adas_state == 'ASSIST':
            # Active Evasion State: Steering take-over + speed capping
            self.angular_vel = self.adas_steer
            max_allowed = max(0.0, min(8.0, self.ttc_value * 1.5))
            if self.linear_vel > max_allowed:
                self.linear_vel = max_allowed
            adas_note = f">>> AUTONOMOUS EVASION ({self.linear_vel:.1f} m/s) <<<"
                
        elif self.adas_state == 'WARNING':
            # Visual/Auditory Warning: User throttle capped, user steering active
            max_allowed = max(0.0, min(10.0, self.ttc_value * 2.0))
            if self.linear_vel > max_allowed:
                self.linear_vel = max_allowed
                adas_note = f">>> WARNING: SLOW DOWN ({max_allowed:.1f} m/s) <<<"
        
        elif self.adas_state == 'MANUAL_ONLY':
            # Soft Intervention: LKA (Lane Keep Assist) nudge if drifting
            if abs(self.angular_vel) < 0.2 and abs(self.adas_steer) > 0.35:
                self.angular_vel = self.angular_vel * 0.4 + self.adas_steer * 0.6
                adas_note = ">>> LKA: PREVENTING OFF-ROAD! <<<"

        # Final Broadcast
        msg = Twist()
        msg.linear.x = self.linear_vel
        msg.angular.z = self.angular_vel
        self.cmd_pub.publish(msg)
        self.current_adas_note = adas_note

    # ── HUD Renderer (Visuals) ─────────────────────────────────────────

    def _build_frame(self):
        """Composes metadata, telemetry, and camera feed into a unified frame."""
        if self.latest_image is None:
            # Placeholder for initialization
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'LINKING SENSORS...', (180, 240), 1, 1.5, (255, 255, 255), 2)
            cv2.putText(frame, 'Click HUD to activate controls', (180, 270), 1, 0.7, (0, 255, 255), 1)
        else:
            frame = self.latest_image.copy()

        h, w = frame.shape[:2]

        # ─── Cinematic Overlays ───
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # ─── Telemetry Engine ───
        # ADAS Metadata
        m_col = (0, 200, 255) if self.adas_mode == 'ml' else (255, 100, 0)
        cv2.putText(frame, f'SYSTEM: {self.adas_mode.upper()}', (15, 25), 1, 1.0, m_col, 2)

        # Hysteresis-Aware Safety State
        s_colors = {
            'MANUAL_ONLY': (50, 255, 50), 'WARNING': (50, 255, 255),
            'ASSIST': (0, 165, 255), 'EMERGENCY_BRAKE': (50, 50, 255)}
        sc = s_colors.get(self.adas_state, (200, 200, 200))
        cv2.putText(frame, f'STATE: {self.adas_state}', (250, 25), 1, 1.0, sc, 2)

        # Speedometer
        speed_disp = self.speed * 3.6
        cv2.putText(frame, f'{speed_disp:3.0f} km/h', (15, 60), 1, 1.5, (255, 255, 255), 2)

        # Safety Metrics
        ttc_str = f'TTC: {self.ttc_value:.1f}s' if self.ttc_value < 100 else 'TTC: N/A'
        cv2.putText(frame, ttc_str, (300, 60), 1, 1.0, (200, 200, 200), 1)
        cv2.putText(frame, f'SCAN: {self.min_distance:.1f}m', (500, 60), 1, 1.0, (200, 200, 200), 1)

        # Virtual Steering Rack
        bar_center = w // 2
        bar_x = int(bar_center + (self.angular_vel / 1.5) * 150)
        cv2.line(frame, (bar_center - 150, 75), (bar_center + 150, 75), (50, 50, 50), 2)
        cv2.circle(frame, (bar_x, 75), 8, (0, 255, 0), -1)

        # ─── Intervention Alerts ───
        if self.adas_state in ['EMERGENCY_BRAKE', 'ASSIST']:
            border_col = (0, 0, 255) if self.adas_state == 'EMERGENCY_BRAKE' else (0, 165, 255)
            cv2.rectangle(frame, (0, 0), (w, h), border_col, 8)
            if hasattr(self, 'current_adas_note') and self.current_adas_note:
                cv2.putText(frame, self.current_adas_note, (w//2-250, h//2 + 80), 1, 1.5, border_col, 3)

        return frame


def main(args=None):
    rclpy.init(args=args)
    # Mode Parsing
    adas_mode = 'fixed'
    if '--mode' in sys.argv:
        idx = sys.argv.index('--mode')
        if idx + 1 < len(sys.argv): adas_mode = sys.argv[idx + 1]
        
    node = ManualDriver(adas_mode=adas_mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
