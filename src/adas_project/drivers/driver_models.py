import random
import numpy as np

class Driver:
    def __init__(self, style):
        self.style = style.upper() # AGGRESSIVE, NORMAL, DEFENSIVE
        self.target_speed = self._get_target_speed()
        self.target_lane = 0.0
        self.intent_lane_change = 0
        self.intent_brake = 0
        self.intent_overtake = 0
        self.intent_offroad = 0
        self.time_in_lane = 0.0
        
    def _get_target_speed(self):
        if self.style == 'AGGRESSIVE': return 25.0
        if self.style == 'NORMAL': return 15.0
        return 10.0 # DEFENSIVE
        
    def get_action(self, state, dt=0.1):
        self.time_in_lane += dt
        
        # Reset intents
        self.intent_lane_change = 0
        self.intent_brake = 0
        self.intent_overtake = 0
        self.intent_offroad = 0
        
        speed = state['speed']
        lane = state['lateral_deviation']
        obs_dist = state['obstacle_distance']
        ttc = state['ttc']
        
        # Base limits
        if self.style == 'AGGRESSIVE':
            ttc_thresh = 1.5
            dist_thresh = 10.0
        elif self.style == 'NORMAL':
            ttc_thresh = 3.0
            dist_thresh = 20.0
        else:
            ttc_thresh = 5.0
            dist_thresh = 40.0
            
        steer = 0.0
        throttle = 0.0
        brake = 0.0
        
        # Speed control
        if obs_dist < dist_thresh or ttc < ttc_thresh:
            self.intent_brake = 1
            brake = 0.5 + 0.5 * (1.0 - (ttc / ttc_thresh)) if ttc > 0 else 1.0
            brake = min(1.0, max(0.0, brake))
            # Overtake logic
            if self.style in ['AGGRESSIVE', 'NORMAL'] and self.time_in_lane > 2.0:
                if random.random() < (0.05 if self.style == 'AGGRESSIVE' else 0.01):
                    self.intent_overtake = 1
                    self.intent_lane_change = 1
                    self.target_lane = 4.0 if lane < 2.0 else 0.0
                    self.time_in_lane = 0.0
        else:
            if speed < self.target_speed:
                throttle = 0.5 + random.uniform(0, 0.2)
            else:
                throttle = 0.1
                brake = 0.1
                
        # Lane keeping
        lane_error = self.target_lane - lane
        steer = 0.1 * lane_error
        
        # Stochastic noise
        if self.style == 'AGGRESSIVE':
            steer += random.gauss(0, 0.05)
            if random.random() < 0.005: # Occasional offroad intent
                self.intent_offroad = 1
                steer += 0.5 * random.choice([-1, 1])
                
        steer = np.clip(steer, -0.5, 0.5)
        
        return steer, throttle, brake, self.intent_lane_change, self.intent_brake, self.intent_overtake, self.intent_offroad
