#!/usr/bin/env python3
"""
scenario_controller.py — Dynamic Traffic & Scenario Orchestrator.

Fixed the missing injection logic.
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ModelState, ModelStates
import math
import time
import random

EGO_NAME        = 'adas_vehicle'
EGO_RADIUS      = 1.0
OBSTACLE_RADIUS = 1.5
SPAWN_GRACE     = 8.0    

TRACK_R   = 48.5
STRAIGHT  = 200.0
SEMI      = TRACK_R * math.pi
TRACK_LEN = 2 * STRAIGHT + 2 * SEMI

COLORS = {
    'moving': '0.8 0.1 0.1 1'
}

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
      <collision name='col_chassis'><pose>0 0 0.4 0 0 0</pose><geometry><box><size>4.5 1.8 1.0</size></box></geometry></collision>
      <collision name='col_cabin'><pose>0.3 0 1.15 0 0 0</pose><geometry><box><size>2.0 1.6 0.6</size></box></geometry></collision>
    </link>
  </model>
</sdf>"""


class ScenarioController(Node):
    def __init__(self):
        super().__init__('scenario_controller')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.model_state_pub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_cb, 10)

        self.ego_x, self.ego_y, self.ego_yaw = -50.0, 46.5, math.pi

        # Obstacles AHEAD of ego in CW direction
        # CCW s=0-100 → top straight x=100→0 (ego drives East toward these)
        # CCW s=550-700 → right curve (ego enters after top straight)
        self.obstacles = [
            self._make('car_1',  'moving', -2.0,  100.0, 2.0),
            self._make('car_2',  'moving',  2.0,  130.0, 2.5),
            self._make('car_3',  'moving', -2.0,  160.0, 1.5),
            self._make('car_4',  'moving', -2.0,  190.0, 2.0),
            self._make('car_5',  'moving',  2.0,  220.0, 3.0),
            self._make('car_6',  'moving', -2.0,  250.0, 2.0),
            self._make('car_7',  'moving', -2.0,  280.0, 1.5),
            self._make('car_8',  'moving',  2.0,  310.0, 2.5),
            self._make('car_9',  'moving', -2.0,  340.0, 2.0),
            self._make('car_10', 'moving', -2.0,  380.0, 1.5),
        ]

        self.get_logger().info('[SCENARIO] Initializing Scene...')
        self.spawn_client.wait_for_service(timeout_sec=10.0)
        self.spawn_obstacles()

        self.timer = self.create_timer(0.1, self.tick)
        self.time_t = 0.0

    def _make(self, name, t, lane, dist, spd):
        return {'name': name, 'type': t, 'lane': lane, 'target_lane': lane,
                'dist_ahead': dist, 'base_speed': spd, 'speed': spd, 's': dist,
                'current_x': 0.0, 'current_y': 0.0,
                'next_action_t': random.uniform(5.0, 12.0),
                'behavior': 'cruise', 'behavior_end_t': 0.0}

    def model_cb(self, msg):
        if EGO_NAME in msg.name:
            idx = msg.name.index(EGO_NAME)
            self.ego_x = msg.pose[idx].position.x
            self.ego_y = msg.pose[idx].position.y

    def get_s_from_xy(self, x, y):
        """Simplistic projection to track S to find distance in CW direction."""
        if y > 0 and -100 <= x <= 100: 
            return x + 100.0
        if y <= 0 and -100 <= x <= 100: 
            return STRAIGHT + SEMI + 100.0 - x
        if x > 100:
            a = math.atan2(y, x - 100.0) # pi/2 down to -pi/2
            return STRAIGHT + (math.pi/2 - a) * TRACK_R
        else:
            a = math.atan2(y, x + 100.0) # -pi/2 to -pi, then pi down to pi/2
            # Handle atan2 wrap-around:
            if a > 0:
                dist_angle = 3*math.pi/2 - a
            else:
                dist_angle = -math.pi/2 - a
            return STRAIGHT + SEMI + STRAIGHT + dist_angle * TRACK_R

    def spawn_obstacles(self):
        for obs in self.obstacles:
            req = SpawnEntity.Request()
            req.name = obs['name']
            req.xml = make_car_sdf(obs['name'], obs['type'])
            req.robot_namespace = obs['name']
            bx, by, byaw = self.get_xy_from_s(obs['s'])
            ox = bx - obs['lane'] * math.sin(byaw)
            oy = by + obs['lane'] * math.cos(byaw)
            req.initial_pose.position.x = ox
            req.initial_pose.position.y = oy
            req.initial_pose.position.z = 0.1
            q = self.yaw_to_quat(byaw)
            req.initial_pose.orientation.z = q.z
            req.initial_pose.orientation.w = q.w
            self.spawn_client.call_async(req)

    def yaw_to_quat(self, yaw):
        class Q: pass
        q = Q()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def get_xy_from_s(self, s):
        s = s % TRACK_LEN
        
        # 1. Top Straight (CW: x from -100 to 100, y = 48.5)
        if s < STRAIGHT:
            return -100.0 + s, TRACK_R, 0.0
            
        s2 = s - STRAIGHT
        # 2. Right Curve (CW: center (100,0), radius 48.5, angle from pi/2 down to -pi/2)
        if s2 < SEMI:
            a = s2 / TRACK_R
            theta = math.pi/2 - a
            return (100.0 + TRACK_R * math.cos(theta),
                    TRACK_R * math.sin(theta),
                    -a)
                    
        s3 = s2 - SEMI
        # 3. Bottom Straight (CW: x from 100 to -100, y = -48.5)
        if s3 < STRAIGHT:
            return 100.0 - s3, -TRACK_R, math.pi
            
        s4 = s3 - STRAIGHT
        # 4. Left Curve (CW: center (-100,0), radius 48.5, angle from -pi/2 down to -3pi/2)
        a = s4 / TRACK_R
        theta = -math.pi/2 - a
        yaw = math.pi - a
        return (-100.0 + TRACK_R * math.cos(theta),
                TRACK_R * math.sin(theta),
                yaw)

    def _update_behavior(self, obs):
        # Reset behavior if elapsed
        if obs['behavior'] != 'cruise' and self.time_t >= obs['behavior_end_t']:
            obs['behavior'] = 'cruise'
            obs['speed'] = obs['base_speed']
            
        if self.time_t < obs['next_action_t']: return
        
        roll = random.random()
        if roll < 0.20:
            obs['behavior'] = 'braking'
            obs['speed'] = obs['base_speed'] * 0.3 
            obs['behavior_end_t'] = self.time_t + random.uniform(1.5, 4.0)
        elif roll < 0.50:
            obs['behavior'] = 'lane_change'
            obs['target_lane'] = 2.0 if obs['lane'] < 0 else -2.0
            obs['behavior_end_t'] = self.time_t + 3.0
        
        obs['next_action_t'] = self.time_t + random.uniform(5.0, 12.0)

    def tick(self):
        dt = 0.1
        self.time_t += dt

        for obs in self.obstacles:
            bx, by, byaw = self.get_xy_from_s(obs['s'])
            ox = bx - obs['lane'] * math.sin(byaw)
            oy = by + obs['lane'] * math.cos(byaw)

            dist = math.sqrt((ox - self.ego_x)**2 + (oy - self.ego_y)**2)
            if self.time_t > SPAWN_GRACE and dist < (EGO_RADIUS + OBSTACLE_RADIUS):
                self.get_logger().error(f'!!! COLLISION DETECTED !!! Ego ↔ {obs["name"]}')

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ScenarioController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
