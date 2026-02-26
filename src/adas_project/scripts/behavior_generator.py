#!/usr/bin/env python3
"""
behavior_generator.py — Smart Reactive Driver with 360° Lidar-Based Decisions.

Track: CLOCKWISE oval.  Top→East, Bottom→West.
Spawn: (-50, 46.5) facing East.

Lane changes use lidar sectors:
  FRONT  → obstacle ahead → trigger lane change
  REAR   → obstacle passed → safe to return to original lane

Driver profiles differ in trigger distances:
  AGGRESSIVE  — late reaction, cuts it close, fast
  NORMAL      — medium reaction, safe margins
  DEFENSIVE   — early reaction, huge margins, slow
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
import math, random, json

TRACK_CENTER = 48.5
RIGHT_LANE   = 46.5
LEFT_LANE    = 50.5


class BehaviorGenerator(Node):
    def __init__(self, mode='auto', driver_type='NORMAL'):
        super().__init__('behavior_generator')
        self.mode = mode
        self.driver_type = driver_type.upper()

        cmd_topic = '/adas/cmd_vel' if self.mode == 'manual' else '/cmd_vel'
        self.cmd_pub       = self.create_publisher(Twist, cmd_topic, 10)
        self.telemetry_pub = self.create_publisher(String, '/driver_telemetry', 10)

        self.create_subscription(LaserScan,   '/scan',                self.scan_cb, 10)
        self.create_subscription(Odometry,    '/odom',                self.odom_cb, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)

        self.create_timer(0.05, self.control_loop)  # 20 Hz

        # ── Lidar sectors ──
        self.front_dist       = 100.0   # Narrow cone ahead
        self.front_left_dist  = 100.0   # Left-front sector
        self.front_right_dist = 100.0   # Right-front sector
        self.right_dist       = 100.0   # Right side
        self.rear_right_dist  = 100.0   # Behind-right
        self.rear_dist        = 100.0   # Directly behind
        self.left_dist        = 100.0   # Left side

        # Model-based closest obstacle
        self.min_obs_dist     = 100.0
        self.dist_ahead_right = 100.0
        self.dist_ahead_left  = 100.0

        # State
        self.steer_cmd = 0.0
        self.velocity  = 0.0
        self.ego_x     = -50.0
        self.ego_y     = 46.5
        self.ego_yaw   = 0.0

        # Lane FSM
        self.target_lane = RIGHT_LANE
        # We only consider the two lanes going in our current direction (CW) to prevent crossing the median
        self.lanes = [RIGHT_LANE, LEFT_LANE]
        self.lc_state = 'LANE_RIGHT'
        self.lc_settle   = 0

        # Intent flags
        self.intent_lane_change = 0
        self.intent_brake       = 0
        self.intent_overtake    = 0
        self.intent_offroad     = 0

        self.tick_count = 0
        self.collision_sticky = 0
        self.get_logger().info(f"[DRIVER] {self.driver_type} — lidar-based reactive")

    # ── Callbacks ────────────────────────────────────────────────────

    def odom_cb(self, msg):
        self.velocity = math.hypot(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y)

    def scan_cb(self, msg):
        n = len(msg.ranges)
        if n == 0:
            return

        def sector_min(start_idx, end_idx):
            """Min distance in a sector, wrapping around."""
            vals = []
            for i in range(start_idx, end_idx):
                r = msg.ranges[i % n]
                if 0.15 < r < 200.0 and not math.isnan(r) and not math.isinf(r):
                    vals.append(r)
            return min(vals) if vals else 100.0

        # Gazebo lidar angle ranges from -pi to pi.
        # ranges[0] is -pi (Rear), ranges[n/2] is 0 (Front), ranges[n-1] is +pi (Rear).
        # We will split the array into 8 chunks of size s.
        s = n // 8
        
        # Sector indices (0 to 8s):
        # 0s-1s: -180 to -135 (Rear Right)
        # 1s-2s: -135 to -90  (Right)
        # 2s-3s: -90  to -45  (Front Right)
        # 3s-4s: -45  to 0    (Front Right-Center)
        # 4s-5s: 0    to 45   (Front Left-Center)
        # 5s-6s: 45   to 90   (Front Left)
        # 6s-7s: 90   to 135  (Left)
        # 7s-8s: 135  to 180  (Rear Left)

        self.rear_right_dist  = sector_min(0, s)
        self.right_dist       = sector_min(s, 2*s)
        self.front_right_dist = sector_min(2*s, 4*s) # combine 2s-4s (-90 to 0)
        
        # Front distance should be highly specific, approx -15 to +15 degrees
        # (255 to 285 out of 540 if split 6 ways... let's just do math on indices)
        c = n // 2
        dw = n // 24 # 15 degrees each way
        self.front_dist       = sector_min(c - dw, c + dw)
        
        self.front_left_dist  = sector_min(4*s, 6*s) # combine 4s-6s (0 to +90)
        self.left_dist        = sector_min(6*s, 7*s)
        self.rear_dist        = min(sector_min(0, s), sector_min(7*s, n)) # combine both rear extremes

    def model_cb(self, msg):
        # We still need ego pose for CTE / lane keeping
        for i, name in enumerate(msg.name):
            if name == 'adas_vehicle':
                self.ego_x = msg.pose[i].position.x
                self.ego_y = msg.pose[i].position.y
                q = msg.pose[i].orientation
                self.ego_yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y**2 + q.z**2))
                break
        
        # NOTE: We no longer cheat by reading other model states to find distances.
        # Everything relies purely on the 360-degree LiDAR sweep.

    # ── Track Geometry (CW) ──────────────────────────────────────────

    def get_track_errors(self, lane_r):
        cx, cy = self.ego_x, self.ego_y
        if cy > 0 and -100 <= cx <= 100:
            target_yaw = 0.0
            cte = cy - lane_r
        elif cy <= 0 and -100 <= cx <= 100:
            target_yaw = math.pi
            cte = -(cy + lane_r)
        elif cx > 100:
            angle = math.atan2(cy, cx - 100.0)
            target_yaw = angle - math.pi / 2.0
            cte = math.hypot(cx - 100.0, cy) - lane_r
        else:
            angle = math.atan2(cy, cx + 100.0)
            target_yaw = angle - math.pi / 2.0
            cte = math.hypot(cx + 100.0, cy) - lane_r
        yaw_err = self.ego_yaw - target_yaw
        while yaw_err >  math.pi: yaw_err -= 2.0 * math.pi
        while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
        return cte, yaw_err

    # ── Driver Parameters ────────────────────────────────────────────

    def _p(self):
        if self.driver_type == 'AGGRESSIVE':
            return {
                'speed':        7.0,    # Cap max speed to retain physical grip
                'avoid_dist':   25.0,   
                'return_dist':  12.0,   
                'brake_dist':   10.0,   
                'slow_dist':    15.0,   
                'noise':        0.005,  # Erratic steering
                'random_lc':    0.002,  # Frequent lane changes
            }
        elif self.driver_type == 'DEFENSIVE':
            return {
                'speed':        4.0,    # Slow and safe
                'avoid_dist':   40.0,   
                'return_dist':  20.0,   
                'brake_dist':   20.0,   
                'slow_dist':    30.0,   
                'noise':        0.001,  
                'random_lc':    0.0,
            }
        else:  # NORMAL
            return {
                'speed':        5.5,
                'avoid_dist':   30.0,
                'return_dist':  15.0,
                'brake_dist':   15.0,
                'slow_dist':    20.0,
                'noise':        0.002,
                'random_lc':    0.001,
            }

    # ── Lidar-Based Lane Change ──────────────────────────────────────

    def update_lane_fsm(self):
        p = self._p()
        self.intent_lane_change = 0
        self.intent_overtake    = 0

        # Settling timer: block new lane changes for 30 ticks after completing one
        if self.lc_settle > 0:
            self.lc_settle -= 1
            return

        if self.lc_state == 'LANE_RIGHT':
            self.target_lane = RIGHT_LANE
            
            # Obstacle directly ahead (narrow 30deg cone)
            if self.front_dist < p['avoid_dist']:
                # Can we merge left? Check left sector (-90 to 0) + rear left (-180 to -135)
                # Need space to slide over safely
                if self.front_left_dist > p['avoid_dist'] * 0.8 and self.left_dist > 5.0:
                    self.lc_state = 'CHANGING_LEFT'
                    self.intent_lane_change = 1
                    self.intent_overtake = 1
            elif random.random() < p['random_lc']:
                if self.front_left_dist > p['avoid_dist'] * 0.8 and self.left_dist > 5.0:
                    self.lc_state = 'CHANGING_LEFT'
                    self.intent_lane_change = 1

        elif self.lc_state == 'CHANGING_LEFT':
            self.target_lane = LEFT_LANE
            self.intent_lane_change = 1
            cte, _ = self.get_track_errors(LEFT_LANE)
            if abs(cte) < 0.5:
                self.lc_state = 'LANE_LEFT'
                self.lc_settle = random.randint(15, 50)

        elif self.lc_state == 'LANE_LEFT':
            self.target_lane = LEFT_LANE
            # Return to right lane if right front/right side/rear right are clear
            if self.front_right_dist > p['return_dist'] and self.right_dist > 4.0 and self.rear_right_dist > p['return_dist'] * 0.5:
                # But don't return if we are about to crash in the left lane (in which case we shouldn't change lanes)
                self.lc_state = 'CHANGING_RIGHT'
                self.intent_lane_change = 1
            # Emergency dodge back right if left is blocked and right just opened up
            elif self.front_dist < p['avoid_dist']:
                if self.front_right_dist > p['return_dist'] * 0.5 and self.right_dist > 4.0:
                    self.lc_state = 'CHANGING_RIGHT'
                    self.intent_lane_change = 1

        elif self.lc_state == 'CHANGING_RIGHT':
            self.target_lane = RIGHT_LANE
            self.intent_lane_change = 1
            cte, _ = self.get_track_errors(RIGHT_LANE)
            if abs(cte) < 0.5:
                self.lc_state = 'LANE_RIGHT'
                self.lc_settle = random.randint(15, 50)

    # ── Velocity ─────────────────────────────────────────────────────

    def compute_vel(self):
        p = self._p()
        
        # We always check the front lidar cone for emergency braking
        fd = self.front_dist 
            
        self.intent_brake = 0

        if fd < p['brake_dist']:
            self.intent_brake = 1
            return 0.5 # Max brake

        if fd < p['slow_dist']:
            self.intent_brake = 1
            return p['speed'] * max(0.3, fd / p['slow_dist'])

        return p['speed']

    # ── Control Loop ─────────────────────────────────────────────────

    def control_loop(self):
        self.tick_count += 1
        self.update_lane_fsm()
        self.intent_offroad = 0

        cte, yaw_err = self.get_track_errors(self.target_lane)

        # Stanley Controller for high-speed stability
        v = max(2.5, self.velocity)
        if self.lc_state in ('CHANGING_LEFT', 'CHANGING_RIGHT'):
            k = 0.20  # Gentle merge
        else:
            k = 0.15  # Smooth tracking

        steer_raw = -yaw_err - math.atan2(k * cte, v)

        # Driver noise (after warmup)
        p = self._p()
        if self.tick_count > 50 and p['noise'] > 0:
            steer_raw += random.gauss(0, p['noise'])

        # Aggressive swerve
        if self.driver_type == 'AGGRESSIVE' and self.tick_count > 50:
            if random.random() < 0.005:
                self.intent_offroad = 1
                steer_raw += random.choice([-0.1, 0.1])

        self.steer_cmd = min(0.2, max(-0.2, steer_raw))
        
        # Hard limits on steering prevent endless circling
        max_steer = 0.2
        self.steer_cmd = max(-max_steer, min(max_steer, self.steer_cmd))

        vel = self.compute_vel()

        # ── Collision detection (model-based only, sticky) ──
        if self.min_obs_dist < 2.5:
            self.collision_sticky = 10
        elif self.collision_sticky > 0:
            self.collision_sticky -= 1

        # ── Telemetry ──
        cte_c, _ = self.get_track_errors(TRACK_CENTER)
        throttle  = min(1.0, max(0.0, (vel - self.velocity) / 5.0))
        brake_val = min(1.0, max(0.0, (self.velocity - vel) / 5.0))
        if self.velocity > vel + 0.5:
            brake_val = 1.0

        state = {
            'speed':              self.velocity,
            'acceleration':       throttle * 3.0 - brake_val * 6.0,
            'lateral_deviation':  cte_c,
            'yaw':                self.ego_yaw,
            'steering':           self.steer_cmd,
            'throttle':           throttle,
            'brake':              brake_val,
            'obstacle_distance':  self.front_dist,
            'ttc':                self.front_dist / (self.velocity + 0.001),
            'offroad':            1 if abs(cte_c) > 6.0 else 0,
            'collision':          int(self.collision_sticky > 0),
            'driver_type':        self.driver_type,
            'intent_lane_change': self.intent_lane_change,
            'intent_brake':       self.intent_brake,
            'intent_overtake':    self.intent_overtake,
            'intent_offroad':     self.intent_offroad,
        }
        tmsg = String()
        tmsg.data = json.dumps(state)
        self.telemetry_pub.publish(tmsg)

        cmd = Twist()
        cmd.linear.x  = vel
        cmd.angular.z = self.steer_cmd
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    import sys
    mode = 'auto'; driver_type = 'NORMAL'
    for i, a in enumerate(sys.argv):
        if a == '--mode'   and i+1 < len(sys.argv): mode = sys.argv[i+1]
        if a == '--driver' and i+1 < len(sys.argv): driver_type = sys.argv[i+1]
    node = BehaviorGenerator(mode=mode, driver_type=driver_type)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
