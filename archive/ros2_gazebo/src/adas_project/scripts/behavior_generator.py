#!/usr/bin/env python3
"""
behavior_generator.py — Ego-Vehicle Intelligent Motion Planner.

This node implements a sophisticated lane-keeping and obstacle-avoidance 
system for the ego-vehicle. It mimics realistic driving by switching 
between different "Driver Policies" (Aggressive, Defensive, etc.).

Key Logic:
- Proactive Overtaking: Uses LiDAR to detect slow obstacles and initiates 
  lane changes if the adjacent lane is clear.
- Smooth PID-based Lane Keeping: Maintains track center using track geometry 
  lookups and error-correcting steering logic.
- Intent Simulation: Randomly switches between driving styles to provide 
  diverse data for the ML inference engine.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import math
import random

# ─── track geometry ──────────────────────────────────────────────────
# ─── Track & Geometry Multi-tier Logic ────────────────────────────────
TRACK_CENTER     = 48.5
RIGHT_LANE       = TRACK_CENTER - 2.0   # 46.5
LEFT_LANE        = TRACK_CENTER + 2.0   # 50.5

# ─── Behavior Thresholds ──────────────────────────────────────────────
LC_TRIGGER_DIST  = 25.0   # Initiate overtake when obstacle is closer than 25m
LC_CLEAR_DIST    = 80.0   # Return to right lane when 80m clear ahead
LC_RIGHT_CLEAR   = 25.0   # Blind-spot check distance for returning
EMERG_STOP_DIST  = 4.0    # Hard-stop safety floor


class BehaviorGenerator(Node):
    """
    ROS2 Node for ego-motion control and behavior simulation.
    
    Generates realistic driving patterns and handles reactive 
    lane-change maneuvers.
    """
    def __init__(self, mode='auto'):
        super().__init__('behavior_generator')

        self.mode = mode
        # Output arbitration: Manual mode sends to a mixable topic, 
        # Auto mode sends directly to the hardware/sim interface.
        cmd_topic = '/adas/cmd_vel' if self.mode == 'manual' else '/cmd_vel'
        
        self.cmd_pub    = self.create_publisher(Twist, cmd_topic, 10)
        self.policy_pub = self.create_publisher(String, '/active_policy', 10)

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)

        # Simulation Metadata: Driving Styles
        self.policies = ['aggressive', 'defensive', 'inconsistent', 'late_braking']
        self.current_policy = random.choice(self.policies)

        # Timers: Policy rotation and High-frequency control (20Hz)
        self.create_timer(5.0,  self.switch_policy)
        self.create_timer(0.05, self.publish_control)

        # 360° Perception State (Aggregated LiDAR sectors)
        self.front_dist       = 100.0
        self.front_left_dist  = 100.0
        self.front_right_dist = 100.0
        self.rear_dist        = 100.0
        self.rear_right_dist  = 100.0
        self.rear_left_dist   = 100.0

        # Motion State
        self.steer_cmd = 0.0
        self.ego_x     = 0.0
        self.ego_y     = 0.0
        self.ego_yaw   = 0.0

        # Lane Decision State Machine
        self.target_lane    = RIGHT_LANE
        self.lc_state       = 'LANE_RIGHT' # [LANE_RIGHT, CHANGING_LEFT, LANE_LEFT, CHANGING_RIGHT]

        self.get_logger().info("Ego Behavior Engine Active. Initiating in RIGHT lane.")

    # ── callbacks ─────────────────────────────────────────────────────

    def get_yaw_from_quat(self, q):
        """Converts Gazebo quaternion to Euler Yaw."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def model_cb(self, msg):
        """Processes ego ground truth pose."""
        if 'adas_vehicle' in msg.name:
            idx = msg.name.index('adas_vehicle')
            self.ego_x   = msg.pose[idx].position.x
            self.ego_y   = msg.pose[idx].position.y
            self.ego_yaw = self.get_yaw_from_quat(msg.pose[idx].orientation)

    def switch_policy(self):
        """Periodically alters the driving style for dataset diversity."""
        self.current_policy = random.choice(self.policies)
        self.get_logger().info(f"[STYLE SWITCH] New Policy: {self.current_policy}")

    def scan_cb(self, msg):
        """Partitions 360-degree LiDAR into functional sectors (Front, Sides, Blindspots)."""
        n = len(msg.ranges)
        if n == 0: return

        def safe_min(ranges):
            """Robust minimum distance finder that ignores infinities and noise."""
            vals = [r for r in ranges if 0.15 < r < 200.0 and not math.isnan(r) and not math.isinf(r)]
            return min(vals) if vals else 100.0

        front = n // 2 # 0 degrees (Center)
        
        # Sector Partitioning Logic:
        # +/- 10 degrees -> Frontal Corridor
        self.front_dist       = safe_min(msg.ranges[front-20 : front+20])
        # 20° to 70° Left -> Left Overtake Lane
        self.front_left_dist  = safe_min(msg.ranges[front+40 : front+140])
        # 20° to 70° Right -> Right Side obstacles
        self.front_right_dist = safe_min(msg.ranges[front-140 : front-40])
        # Rear +/- 20° -> Following vehicles
        self.rear_dist        = safe_min(list(msg.ranges[:40]) + list(msg.ranges[-40:]))
        # Blindspots for lane merges
        self.rear_right_dist  = safe_min(msg.ranges[front-320 : front-220])
        self.rear_left_dist   = safe_min(msg.ranges[front+220 : front+320])

    # ── Track Navigation Math ─────────────────────────────────────────

    def get_track_errors(self, target_lane_y):
        """
        Calculates Cross-Track Error (CTE) and Heading Error relative to the road.
        
        Handles:
        1. Straightaways (X-parallel)
        2. Semi-circular banked curves (radial coordinates)
        """
        cx, cy = self.ego_x, self.ego_y

        if cy > 0 and -100 <= cx <= 100:      # Top Straight (Driving East)
            target_yaw = 0.0
            cte = cy - target_lane_y
        elif cy <= 0 and -100 <= cx <= 100:    # Bottom Straight (Driving West)
            target_yaw = math.pi
            cte = -(cy + target_lane_y)
        elif cx > 100:                        # East Curve
            angle = math.atan2(cy, cx - 100.0)
            target_yaw = angle - math.pi / 2
            dist = math.sqrt((cx - 100.0) ** 2 + cy ** 2)
            cte = dist - target_lane_y
        else:                                 # West Curve
            angle = math.atan2(cy, cx + 100.0)
            target_yaw = angle + math.pi / 2
            dist = math.sqrt((cx + 100.0) ** 2 + cy ** 2)
            cte = dist - target_lane_y

        # Normalize yaw error to [-pi, pi]
        yaw_err = self.ego_yaw - target_yaw
        while yaw_err >  math.pi: yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi
        return cte, yaw_err

    # ── Lane-Change Finite State Machine ───────────────────────────────

    def update_lane_fsm(self):
        """Executes the high-level decision logic for overtaking and lane keeping."""
        fd  = self.front_dist
        fld = self.front_left_dist

        prev_state = self.lc_state

        # Gating Logic: If we are in manual override mode, we disable autonomous 
        # lane changes but provide steering-assist to the closest lane.
        if self.mode == 'manual':
            self.target_lane = RIGHT_LANE if abs(self.ego_y - RIGHT_LANE) < abs(self.ego_y - LEFT_LANE) else LEFT_LANE
            return

        # ── State transitions ──
        if self.lc_state == 'LANE_RIGHT':
            self.target_lane = RIGHT_LANE
            # If blocked and left is clear -> Change Left
            if fd < LC_TRIGGER_DIST and fld > LC_TRIGGER_DIST:
                self.lc_state = 'CHANGING_LEFT'
                self.get_logger().warn(f"[OVERTAKE] Initiating Left Lane Change (Obs: {fd:.1f}m)")

        elif self.lc_state == 'CHANGING_LEFT':
            self.target_lane = LEFT_LANE
            cte, _ = self.get_track_errors(LEFT_LANE)
            if abs(cte) < 0.5: # Sufficiently merged
                self.lc_state = 'LANE_LEFT'

        elif self.lc_state == 'LANE_LEFT':
            self.target_lane = LEFT_LANE
            # Return logic: Is the right lane clear (Front & Blindspot)?
            right_lane_clear = self.front_right_dist > LC_RIGHT_CLEAR and self.rear_right_dist > LC_RIGHT_CLEAR
            if right_lane_clear and (fd > LC_CLEAR_DIST or fd < LC_TRIGGER_DIST):
                self.lc_state = 'CHANGING_RIGHT'

        elif self.lc_state == 'CHANGING_RIGHT':
            self.target_lane = RIGHT_LANE
            cte, _ = self.get_track_errors(RIGHT_LANE)
            if abs(cte) < 0.5:
                self.lc_state = 'LANE_RIGHT'

        if prev_state != self.lc_state:
            self.get_logger().info(f"[LANE FSM] {prev_state} ➔ {self.lc_state}")

    # ── Velocity Policy Engine ───────────────────────────────────────────

    def nominal_vel(self):
        """Returns cruising speed based on current simulated driver style."""
        mapping = {'aggressive': 12.0, 'defensive': 6.0, 'late_braking': 10.0}
        return mapping.get(self.current_policy, random.uniform(4.0, 10.0))

    def compute_vel(self):
        """Calculates reactive velocity based on distance to lead vehicle."""
        nom = self.nominal_vel()
        fd  = self.front_dist

        # Don't stop during an active steering maneuver (overtaking)
        if self.lc_state in ('CHANGING_LEFT', 'CHANGING_RIGHT'):
            return nom * 0.75

        # Safety Fallback: Emergency Hard Braking
        if fd < EMERG_STOP_DIST:
            return 0.0
            
        # Proportional Slowdown for approaching traffic
        if fd < LC_TRIGGER_DIST:
            return nom * max(0.2, fd / LC_TRIGGER_DIST)

        return nom

    # ── Master Control Loop ──────────────────────────────────────────────

    def publish_control(self):
        """Evaluates FSM and Geometry to publish Twist commands (Ackermann-like)."""
        # Broadcast Style Metadata
        pmsg = String(data=self.current_policy)
        self.policy_pub.publish(pmsg)

        # Update Decision State
        self.update_lane_fsm()

        # Get geometric errors
        cte, yaw_err = self.get_track_errors(self.target_lane)

        # PD Control for Steering:
        # We increase gains significantly during 'CHANGING' states for crisp maneuvers.
        if self.lc_state in ('CHANGING_LEFT', 'CHANGING_RIGHT'):
            steer_raw = -0.30 * cte - 0.9 * yaw_err
        else:
            steer_raw = -0.08 * cte - 0.6 * yaw_err

        # Smoothing and Saturation (Physical Steering Angle Limits)
        self.steer_cmd = self.steer_cmd * 0.70 + steer_raw * 0.30
        self.steer_cmd = max(min(self.steer_cmd, 0.55), -0.55)

        # Compute Throttle
        vel = self.compute_vel()

        # Telemetry
        if int(time.time() * 2) % 10 == 0: # Throttled verbose log
            self.get_logger().debug(f"[TELEMETRY] Y:{self.ego_y:5.2f} S:{self.lc_state} V:{vel:.1f}")

        # Assemble and Publish
        cmd = Twist()
        cmd.linear.x = vel
        cmd.angular.z = self.steer_cmd
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    import sys
    mode = 'auto'
    for i, arg in enumerate(sys.argv):
        if arg == '--mode' and i+1 < len(sys.argv):
            mode = sys.argv[i+1]
    node = BehaviorGenerator(mode=mode)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
