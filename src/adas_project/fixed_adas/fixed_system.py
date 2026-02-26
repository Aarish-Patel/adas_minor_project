class FixedADAS:
    def __init__(self, level):
        self.level = level
        self.target_speed = 15.0 # default cruise speed
        
    def get_action(self, state, user_steer, user_throttle, user_brake):
        # Base overrides
        steer = user_steer
        throttle = user_throttle
        brake = user_brake
        
        speed = state['speed']
        ttc = state['ttc']
        lane = state['lateral_deviation']
        obs_dist = state['obstacle_distance']
        
        # Level 1: Speed maintain
        if self.level >= 1:
            if speed < self.target_speed and obs_dist > 30.0:
                throttle = max(throttle, 0.4)
                
        # Level 2: Lane keeping + emergency stop
        if self.level >= 2:
            # Hard safety limits
            if abs(lane) > 3.0:
                steer = -0.15 * lane # Proportional correction
            
            if ttc < 2.0 or obs_dist < 8.0:
                brake = 1.0
                throttle = 0.0
                
        # Level 3: Obstacle avoidance path planning
        if self.level >= 3:
            if ttc < 3.5 and obs_dist < 25.0:
                # Steer away if blocked
                steer = 0.4 if lane < 2.0 else -0.4
                
        # Level 4: Full autonomy (Optimal path)
        if self.level == 4:
            # Full override of user inputs
            steer = -0.1 * lane
            
            if obs_dist < 15.0 or ttc < 2.5:
                brake = 0.8
                throttle = 0.0
            else:
                throttle = 0.6 if speed < self.target_speed else 0.0
                brake = 0.0
                
            # Overtake autonomously if safe
            if obs_dist < 30.0 and speed > 5.0:
                steer = 0.2 if lane < 2.0 else -0.2
                
        # Hard Safety Override ALWAYS dominates for Level 2+
        if self.level >= 2:
            if ttc < 1.0 or obs_dist < 5.0:
                brake = 1.0
                throttle = 0.0
            if abs(lane) > 5.0:
                steer = -0.3 * lane # Hard yank back to track
                
        return steer, throttle, brake
