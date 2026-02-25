#!/usr/bin/env python3
"""
traffic_behavior_node.py — Realistic traffic behavior generator.

Each obstacle runs one of 5 behavior types with randomized parameters:
  CRUISING      - constant speed
  SLOW_AHEAD    - gradual deceleration
  BRAKE_CHECK   - cruise → sudden brake → resume
  CUT_IN        - drift into ego lane then back
  OVERTAKE      - fast pass in left lane

Publishes /traffic/status with per-obstacle behavior state.
Moves obstacles via SetEntityState using a kinematic model (s, v, a).
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import ModelStates, EntityState
from std_msgs.msg import String
import math
import random
import time
import json

TRACK_CENTER = 48.5
RIGHT_LANE   = -2.0    # lane offset from centerline
LEFT_LANE    =  2.0

BEHAVIORS = ['CRUISING', 'SLOW_AHEAD', 'BRAKE_CHECK', 'CUT_IN', 'OVERTAKE']


class TrafficVehicle:
    """Kinematic model for one obstacle vehicle."""

    def __init__(self, name, behavior, lane, init_s, params):
        self.name     = name
        self.behavior = behavior
        self.lane     = lane
        self.target_lane = lane
        self.s        = init_s
        self.v        = params.get('cruise_speed', 5.0)
        self.a        = 0.0
        self.max_v    = params.get('max_v', 12.0)
        self.min_v    = params.get('min_v', 0.0)
        self.params   = params
        self.phase    = 'cruise'
        self.phase_timer = 0.0

    def update(self, dt, ego_s, L):
        self.phase_timer += dt

        if self.behavior == 'CRUISING':
            self.a = 0.0

        elif self.behavior == 'SLOW_AHEAD':
            # Gradual deceleration over 10s, then coast
            if self.phase == 'cruise' and self.phase_timer > self.params.get('delay', 8.0):
                self.phase = 'slowing'
                self.phase_timer = 0.0
            if self.phase == 'slowing':
                self.a = -self.params.get('decel', 0.5)
                if self.v <= self.params.get('target_speed', 2.0):
                    self.a = 0.0
                    self.v = self.params.get('target_speed', 2.0)
                    self.phase = 'slow_cruise'

        elif self.behavior == 'BRAKE_CHECK':
            brake_at = self.params.get('brake_at', 12.0)
            resume_after = self.params.get('resume_after', 3.0)
            if self.phase == 'cruise' and self.phase_timer > brake_at:
                self.phase = 'braking'
                self.phase_timer = 0.0
                self.a = -self.params.get('brake_decel', 4.0)
            elif self.phase == 'braking':
                if self.v <= 0.5:
                    self.v = 0.5
                    self.a = 0.0
                if self.phase_timer > resume_after:
                    self.phase = 'resuming'
                    self.phase_timer = 0.0
                    self.a = self.params.get('resume_accel', 2.0)
            elif self.phase == 'resuming':
                if self.v >= self.params.get('cruise_speed', 5.0):
                    self.v = self.params.get('cruise_speed', 5.0)
                    self.a = 0.0
                    self.phase = 'cruise'
                    self.phase_timer = 0.0

        elif self.behavior == 'CUT_IN':
            cut_at = self.params.get('cut_at', 10.0)
            hold_in = self.params.get('hold_in', 4.0)
            if self.phase == 'cruise' and self.phase_timer > cut_at:
                self.phase = 'cutting_in'
                self.phase_timer = 0.0
                self.target_lane = RIGHT_LANE if self.lane == LEFT_LANE else LEFT_LANE
            elif self.phase == 'cutting_in':
                # Drift lane
                diff = self.target_lane - self.lane
                self.lane += 0.3 * diff * dt * 10  # smooth transition
                if abs(self.lane - self.target_lane) < 0.1:
                    self.lane = self.target_lane
                    self.phase = 'holding'
                    self.phase_timer = 0.0
            elif self.phase == 'holding' and self.phase_timer > hold_in:
                self.phase = 'returning'
                self.target_lane = self.params.get('home_lane', RIGHT_LANE)
                self.phase_timer = 0.0
            elif self.phase == 'returning':
                diff = self.target_lane - self.lane
                self.lane += 0.3 * diff * dt * 10
                if abs(self.lane - self.target_lane) < 0.1:
                    self.lane = self.target_lane
                    self.phase = 'cruise'
                    self.phase_timer = 0.0

        elif self.behavior == 'OVERTAKE':
            self.a = 0.0  # constant high speed in left lane

        # Kinematic integration
        self.v = max(self.min_v, min(self.max_v, self.v + self.a * dt))
        self.s += self.v * dt

        # Respawn logic
        rel = (self.s - ego_s) % L
        if rel > L / 2:
            rel -= L
        if rel > 250.0:
            self.s = ego_s - 50.0
            self._reset_phase()
        elif rel < -150.0:
            self.s = ego_s + 150.0
            self._reset_phase()

    def _reset_phase(self):
        self.phase = 'cruise'
        self.phase_timer = 0.0
        self.v = self.params.get('cruise_speed', 5.0)
        self.a = 0.0
        self.lane = self.params.get('home_lane', RIGHT_LANE)

    def to_dict(self):
        return {
            'name': self.name,
            'behavior': self.behavior,
            'phase': self.phase,
            'speed': round(self.v, 2),
            'accel': round(self.a, 2),
            'lane': round(self.lane, 2),
        }


class TrafficBehaviorNode(Node):
    def __init__(self):
        super().__init__('traffic_behavior_node')
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)
        self.status_pub = self.create_publisher(String, '/traffic/status', 10)

        self.ego_x = 0.0
        self.ego_y = 0.0

        # Create traffic vehicles with randomized behaviors
        self.vehicles = self._create_traffic()
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info(f'TrafficBehaviorNode: {len(self.vehicles)} vehicles initialized')

    def _create_traffic(self):
        vehicles = []
        configs = [
            ('obst_s1', 'CRUISING',    RIGHT_LANE,  80.0, {'cruise_speed': 0.0, 'max_v': 0.0}),
            ('obst_s2', 'CRUISING',    RIGHT_LANE, 220.0, {'cruise_speed': 0.0, 'max_v': 0.0}),
            ('obst_s3', 'CRUISING',    RIGHT_LANE, 370.0, {'cruise_speed': 0.0, 'max_v': 0.0}),
            ('obst_m1', 'SLOW_AHEAD',  RIGHT_LANE, 130.0, {
                'cruise_speed': 6.0, 'decel': 0.4, 'target_speed': 2.0, 'delay': 10.0}),
            ('obst_m2', 'BRAKE_CHECK', RIGHT_LANE, 180.0, {
                'cruise_speed': 5.0, 'brake_at': 12.0, 'brake_decel': 4.0,
                'resume_after': 3.0, 'resume_accel': 2.0}),
            ('obst_m3', 'CRUISING',    RIGHT_LANE, 280.0, {'cruise_speed': 6.0}),
            ('obst_m4', 'CUT_IN',      LEFT_LANE,  340.0, {
                'cruise_speed': 5.0, 'cut_at': 8.0, 'hold_in': 4.0,
                'home_lane': LEFT_LANE}),
            ('obst_m5', 'SLOW_AHEAD',  RIGHT_LANE, 420.0, {
                'cruise_speed': 5.5, 'decel': 0.3, 'target_speed': 1.5, 'delay': 15.0}),
            ('obst_med', 'CRUISING',   RIGHT_LANE, 460.0, {'cruise_speed': 8.0}),
            ('obst_over', 'OVERTAKE',  LEFT_LANE,  -30.0, {'cruise_speed': 16.0, 'max_v': 18.0}),
        ]
        for name, beh, lane, dist, params in configs:
            params.setdefault('home_lane', lane)
            params.setdefault('max_v', 12.0)
            params.setdefault('min_v', 0.0)
            params.setdefault('cruise_speed', 5.0)
            v = TrafficVehicle(name, beh, lane, dist, params)
            vehicles.append(v)
        return vehicles

    def model_cb(self, msg):
        if 'adas_vehicle' in msg.name:
            idx = msg.name.index('adas_vehicle')
            self.ego_x = msg.pose[idx].position.x
            self.ego_y = msg.pose[idx].position.y

    def get_s_from_xy(self, x, y):
        if y > 0 and -100 <= x <= 100:
            return 100.0 - x
        elif x > 100:
            angle = math.atan2(y, x - 100)
            if angle < 0:
                angle += 2 * math.pi
            return 200.0 + TRACK_CENTER * angle
        elif y <= 0 and -100 <= x <= 100:
            return 200.0 + TRACK_CENTER * math.pi + (x + 100.0)
        else:
            angle = math.atan2(y, x + 100)
            if angle > 0:
                angle -= 2 * math.pi
            return 400.0 + TRACK_CENTER * math.pi + TRACK_CENTER * (-angle)

    def get_xy_from_s(self, s):
        L_str = 200.0
        L_semi = TRACK_CENTER * math.pi
        L = 2 * L_str + 2 * L_semi
        s = s % L
        if s < L_str:
            return 100.0 - s, 0.0, math.pi
        elif s < L_str + L_semi:
            a = (s - L_str) / TRACK_CENTER
            return 100.0 + TRACK_CENTER * math.cos(a), -TRACK_CENTER * math.sin(a), math.pi + a
        elif s < 2 * L_str + L_semi:
            d = s - L_str - L_semi
            return -100.0 + d, 0.0, 0.0
        else:
            a = (s - 2 * L_str - L_semi) / TRACK_CENTER
            return -100.0 + TRACK_CENTER * math.cos(math.pi + a), -TRACK_CENTER * math.sin(math.pi + a), a

    def tick(self):
        L = 400.0 + 97.0 * math.pi
        s_ego = self.get_s_from_xy(self.ego_x, self.ego_y)

        statuses = []
        for v in self.vehicles:
            v.update(0.1, s_ego, L)

            # Compute world pose
            bx, by, byaw = self.get_xy_from_s(v.s)
            ox = bx - v.lane * math.sin(byaw)
            oy = by + v.lane * math.cos(byaw)

            # Teleport via SetEntityState
            state = EntityState()
            state.name = v.name
            state.pose.position.x = ox
            state.pose.position.y = oy
            state.pose.position.z = 0.0
            q_z = math.sin(byaw / 2.0)
            q_w = math.cos(byaw / 2.0)
            state.pose.orientation.z = q_z
            state.pose.orientation.w = q_w
            if self.set_state_client.service_is_ready():
                req = SetEntityState.Request()
                req.state = state
                self.set_state_client.call_async(req)

            statuses.append(v.to_dict())

        # Publish traffic status
        msg = String()
        msg.data = json.dumps(statuses)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficBehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
