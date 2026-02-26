import numpy as np
import math

TRACK_R = 48.5
STRAIGHT = 200.0
SEMI = TRACK_R * math.pi
TRACK_LEN = 2 * STRAIGHT + 2 * SEMI

class FastSim:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.reset()
        
    def reset(self):
        self.time = 0.0
        # Ego state
        self.ego_s = 0.0
        self.ego_v = 5.0
        self.ego_lane = 0.0 # offset from center
        self.ego_steer = 0.0
        self.ego_accel = 0.0
        
        self.collision = False
        self.offroad = False
        
        # Obstacles: list of dicts {s, v, lane}
        self.obstacles = [
            {'s': 50.0, 'v': 8.0, 'lane': 0.0},
            {'s': 120.0, 'v': 10.0, 'lane': 4.0},
            {'s': 200.0, 'v': 6.0, 'lane': 0.0},
            {'s': 300.0, 'v': 9.0, 'lane': 4.0},
        ]
        
    def get_track_state(self):
        # Return dict of telemetry
        return {
            'speed': self.ego_v,
            'acceleration': self.ego_accel,
            'lateral_deviation': self.ego_lane,
            'steering': self.ego_steer,
            'ttc': self.get_ttc(),
            'obstacle_distance': self.get_min_distance(),
            'collision': self.collision,
            'offroad': self.offroad
        }
        
    def get_min_distance(self):
        min_dist = 999.0
        for obs in self.obstacles:
            if abs(obs['lane'] - self.ego_lane) < 2.0: # Same lane roughly
                dist = (obs['s'] - self.ego_s) % TRACK_LEN
                if dist < min_dist and dist > 0:
                    min_dist = dist
        return min_dist

    def get_ttc(self):
        min_ttc = 999.0
        for obs in self.obstacles:
            if abs(obs['lane'] - self.ego_lane) < 2.0:
                dist = (obs['s'] - self.ego_s) % TRACK_LEN
                rel_v = self.ego_v - obs['v']
                if rel_v > 0 and dist > 0:
                    ttc = dist / rel_v
                    if ttc < min_ttc:
                        min_ttc = ttc
        return min_ttc
        
    def step(self, steering, throttle, brake):
        if self.collision:
            return self.get_track_state()
            
        self.time += self.dt
        
        # Physics update
        self.ego_steer = steering
        self.ego_accel = throttle * 3.0 - brake * 6.0
        
        # Drag
        self.ego_accel -= 0.05 * self.ego_v
        
        self.ego_v += self.ego_accel * self.dt
        self.ego_v = max(0.0, min(self.ego_v, 30.0))
        
        self.ego_s = (self.ego_s + self.ego_v * self.dt) % TRACK_LEN
        self.ego_lane += self.ego_v * math.sin(steering) * self.dt
        
        # Obstacles update
        for obs in self.obstacles:
            obs['s'] = (obs['s'] + obs['v'] * self.dt) % TRACK_LEN
            
        # Check collision
        min_dist = self.get_min_distance()
        if min_dist < 4.5: # Vehicle length + margin
            self.collision = True
            
        # Check offroad
        if abs(self.ego_lane) > 6.0:
            self.offroad = True
            
        return self.get_track_state()
