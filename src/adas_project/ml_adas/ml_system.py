class MLADAS:
    def __init__(self, level, models_dict=None):
        self.level = level
        self.target_speed = 15.0
        self.models = models_dict # dict of loaded pickle models
        
    def _predict_intents(self, state):
        if not self.models:
            return 0, 0, 0, 0 # Fallback
            
        import numpy as np
        features = np.array([[state['speed'], state['acceleration'], state['lateral_deviation'], 0.0, 
                             state['steering'], 0.0, 0.0, state['obstacle_distance'], state['ttc']]])
                             
        # Simple extraction
        # Real impl matches ML features. For this test:
        try:
            lc = self.models['intent_lane_change'].predict(features)[0]
            br = self.models['intent_brake'].predict(features)[0]
            ov = self.models['intent_overtake'].predict(features)[0]
            of = self.models['intent_offroad'].predict(features)[0]
            return lc, br, ov, of
        except:
            return 0, 0, 0, 0
            
    def get_action(self, state, user_steer, user_throttle, user_brake):
        steer = user_steer
        throttle = user_throttle
        brake = user_brake
        
        speed = state['speed']
        ttc = state['ttc']
        lane = state['lateral_deviation']
        obs_dist = state['obstacle_distance']
        
        # Predict driver intent
        p_lc, p_br, p_ov, p_of = self._predict_intents(state)
        
        # Adaptive Thresholds
        ttc_thresh = 2.0
        lane_thresh = 3.0
        obs_thresh = 8.0
        
        # If ML detects intent to overtake or lane change, loosen the lane threshold
        if p_lc > 0.5 or p_ov > 0.5:
            lane_thresh = 5.5 # Allow wide lane maneuvers
            
        # If ML detects intention to brake, prepare systems (lower throttle)
        if p_br > 0.5:
            throttle *= 0.5
            
        # If ML detects offroad risk, tighten steering bounds
        if p_of > 0.5:
            lane_thresh = 1.5
            
        # Level 1: Speed maintain
        if self.level >= 1:
            if speed < self.target_speed and obs_dist > 30.0:
                throttle = max(throttle, 0.4)
                
        # Level 2: Lane keeping + emergency stop
        if self.level >= 2:
            if abs(lane) > lane_thresh:
                steer = -0.15 * lane
            
            if ttc < ttc_thresh or obs_dist < obs_thresh:
                brake = 1.0
                throttle = 0.0
                
        # Level 3: Obstacle avoidance
        if self.level >= 3:
            if ttc < (ttc_thresh + 1.5) and obs_dist < (obs_thresh + 15.0):
                # If driver already showing intent to lane change, do not fight them (less interruption)
                if p_lc < 0.5 and p_ov < 0.5:
                    steer = 0.4 if lane < 2.0 else -0.4 
                    
        # Level 4: Full autonomy
        if self.level == 4:
            if p_lc < 0.5: # Follow lane unless changing
                steer = -0.1 * lane
            
            if obs_dist < 15.0 or ttc < 2.5:
                # If overtale intent, smooth steer, otherwise brake
                if p_ov > 0.5:
                    steer = 0.2 if lane < 2.0 else -0.2
                    brake = 0.2
                else:    
                    brake = 0.8
                    throttle = 0.0
            else:
                throttle = 0.6 if speed < self.target_speed else 0.0
                brake = 0.0
                
        # Hard Safety Override ALWAYS dominates for Level 2+
        if self.level >= 2:
            if ttc < 1.0 or obs_dist < 5.0:
                brake = 1.0
                throttle = 0.0
            if abs(lane) > 5.5:
                steer = -0.3 * lane
                
        return steer, throttle, brake
