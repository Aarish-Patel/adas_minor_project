#!/usr/bin/env python3
"""
scenario_controller.py — Dynamic Traffic & Scenario Orchestrator.

This node populates the Gazebo simulation with realistic traffic actors. 
Unlike static obstacles, these vehicles exhibit stochastic behaviors, 
mimicking unpredictable highway scenarios.

Key Functions:
- Dynamic Spawning: Injects entities with custom SDF descriptions.
- Behavior Modeling: Traffic cars randomly change lanes, brake, or cruise.
- Proximity Gating: Prevents traffic from spawning directly on top of the ego.
- Safety Telemetry: Monitors and logs collisions between ego and actors.
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ModelStates, ModelState
import math
import time
import random

# Simulation Tuning
EGO_NAME        = 'adas_vehicle'
EGO_RADIUS      = 2.0
OBSTACLE_RADIUS = 2.5
SPAWN_GRACE     = 8.0    # Seconds to ignore safety checks during init
STALE_TIMEOUT   = 0.5    # Drop updates if latency is too high

# Track Definition (Frenet-like Coordinate System)
TRACK_R   = 48.5
STRAIGHT  = 200.0
SEMI      = TRACK_R * math.pi
TRACK_LEN = 2 * STRAIGHT + 2 * SEMI

COL_MOVING = '0.8 0.1 0.1 1' # Red for active actors
COL_STATIC = '0.2 0.2 0.8 1' # Blue for stationary hazards

def make_car_sdf(name, obs_type='moving'):
    c = COLORS.get(obs_type, '0.8 0.1 0.1 1')
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='{name}'>
    <static>true</static>
    <link name='body'>
      <visual name='chassis'><pose>0 0 0.4 0 0 0</pose><geometry><box><size>4.5 1.8 1.0</size></box></geometry><material><ambient>{c}</ambient><diffuse>{c}</diffuse></material></visual>
      <visual name='cabin'><pose>0.3 0 1.15 0 0 0</pose><geometry><box><size>2.0 1.6 0.6</size></box></geometry><material><ambient>0.15 0.15 0.2 1</ambient></material></visual>
      <visual name='fl'><pose>1.4 0.95 0.3 1.5708 0 0</pose><geometry><cylinder><radius>0.3</radius><length>0.2</length></cylinder></geometry><material><ambient>0.1 0.1 0.1 1</ambient></material></visual>
      <visual name='fr'><pose>1.4 -0.95 0.3 1.5708 0 0</pose><geometry><cylinder><radius>0.3</radius><length>0.2</length></cylinder></geometry><material><ambient>0.1 0.1 0.1 1</ambient></material></visual>
      <visual name='rl'><pose>-1.4 0.95 0.3 1.5708 0 0</pose><geometry><cylinder><radius>0.3</radius><length>0.2</length></cylinder></geometry><material><ambient>0.1 0.1 0.1 1</ambient></material></visual>
      <visual name='rr'><pose>-1.4 -0.95 0.3 1.5708 0 0</pose><geometry><cylinder><radius>0.3</radius><length>0.2</length></cylinder></geometry><material><ambient>0.1 0.1 0.1 1</ambient></material></visual>
      <collision name='col'><pose>0 0 0.5 0 0 0</pose><geometry><box><size>4.5 1.8 1.2</size></box></geometry></collision>
    </link>
  </model>
</sdf>"""


class ScenarioController(Node):
    """
    ROS2 Node for managing Gazebo traffic and physics overrides.
    """
    def __init__(self):
        super().__init__('scenario_controller')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.model_state_pub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)

        # ─── World Metadata ───
        self.ego_x, self.ego_y, self.ego_yaw = -50.0, 46.5, math.pi
        self.ego_stamp = 0.0
        self.obs_positions = {}

        # ─── Traffic Fleet Definition ───
        # Configuring an initial fleet of 7 cars distributed across the track
        self.obstacles = [
            self._make('car_1', 'moving', -2.0, 100.0, 5.0),
            self._make('car_2', 'moving', -2.0, 180.0, 4.5),
            self._make('car_3', 'moving', -2.0, 280.0, 6.0),
            self._make('car_4', 'moving', -2.0, 380.0, 5.5),
            self._make('car_5', 'moving', -2.0, 480.0, 4.0),
            self._make('car_6', 'moving',  2.0, 550.0, 7.0),
            self._make('car_7', 'moving', -2.0, 650.0, 5.0),
        ]
        self.obstacle_names = [o['name'] for o in self.obstacles]

        self.get_logger().info('[SCENARIO] Initializing Scene...')
        self.spawn_client.wait_for_service(timeout_sec=10.0)
        self.spawn_obstacles()

        # Orchestration Loop (10Hz)
        self.timer = self.create_timer(0.1, self.tick)
        self.time_t = 0.0

    def _make(self, name, t, lane, dist, spd):
        """Helper to create actor configuration objects."""
        return {'name': name, 'type': t, 'lane': lane, 'target_lane': lane,
                'dist_ahead': dist, 'base_speed': spd, 'speed': spd, 's': 0.0,
                'current_x': 0.0, 'current_y': 0.0,
                'next_action_t': random.uniform(5.0, 12.0),
                'behavior': 'cruise', 'behavior_end_t': 0.0}

    # ── Geometry Math ─────────────────────────────────────────────────

    def get_xy_from_s(self, s):
        """Converts Track-Relative S (distance along track) to World X,Y."""
        s = s % TRACK_LEN
        if s < STRAIGHT:
            return 100.0 - s, TRACK_R, math.pi
        s2 = s - STRAIGHT
        if s2 < SEMI:
            a = s2 / TRACK_R
            return (-100.0 + TRACK_R * math.cos(math.pi/2 - a),
                    TRACK_R * math.sin(math.pi/2 - a), math.pi + a)
        s3 = s2 - SEMI
        if s3 < STRAIGHT:
            return -100.0 + s3, -TRACK_R, 0.0
        s4 = s3 - STRAIGHT
        a = s4 / TRACK_R
        return (100.0 + TRACK_R * math.cos(-math.pi/2 + a),
                TRACK_R * math.sin(-math.pi/2 + a), a)

    # ── Behavior FSM ──────────────────────────────────────────────────

    def _update_behavior(self, obs):
        """Stochastically alters the state of a traffic agent."""
        if self.time_t < obs['next_action_t']: return

        # Behavioral Logic:
        # 20% Brake suddenly, 20% Switch lanes, 15% Cruise faster, 45% Maintain
        roll = random.random()
        if roll < 0.20:
            obs['behavior'] = 'braking'
            obs['speed'] = obs['base_speed'] * 0.3 # Slow down to 30%
            obs['behavior_end_t'] = self.time_t + random.uniform(1.5, 4.0)
        elif roll < 0.40:
            obs['behavior'] = 'lane_change'
            obs['target_lane'] = 2.0 if obs['lane'] < 0 else -2.0
            obs['behavior_end_t'] = self.time_t + 3.0
        
        obs['next_action_t'] = self.time_t + random.uniform(5.0, 12.0)

    # ── Core Orchestration ─────────────────────────────────────────────

    def tick(self):
        """Main Orchestration Loop: Advances physics and checks for collisions."""
        dt = 0.1
        self.time_t += dt
        s_ego = self.get_s_from_xy(self.ego_x, self.ego_y)

        for obs in self.obstacles:
            self._update_behavior(obs)

            # Advance agent along track using constant-velocity model
            obs['s'] = (obs['s'] + obs['speed'] * dt) % TRACK_LEN

            # Convert to World coordinates and teleport via Gazebo Internal API
            # Note: We use SetModelState instead of force-physics for better stability 
            # in large-scale multi-actor simulations.
            bx, by, byaw = self.get_xy_from_s(obs['s'])
            ox = bx - obs['lane'] * math.sin(byaw)
            oy = by + obs['lane'] * math.cos(byaw)
            
            state = ModelState()
            state.model_name = obs['name']
            state.pose.position.x, state.pose.position.y = ox, oy
            q = self.yaw_to_quat(byaw)
            state.pose.orientation.z, state.pose.orientation.w = q.z, q.w
            self.model_state_pub.publish(state)

            # Collision Logic: Euclidean check between bounding circles
            dist = math.sqrt((ox - self.ego_x)**2 + (oy - self.ego_y)**2)
            if self.time_t > SPAWN_GRACE and dist < (EGO_RADIUS + OBSTACLE_RADIUS):
                self.get_logger().error(f'!!! COLLISION DETECTED !!! Ego ↔ {obs["name"]}')


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ScenarioController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
