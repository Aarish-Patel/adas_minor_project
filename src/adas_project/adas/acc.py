"""Adaptive cruise / follow-the-leader.

The driver's throttle sets the speed they would like; when a slower car is ahead in
the lane the speed is reduced to hold a time-gap behind it. The lead car's speed
comes from the tracker (its own motion, with the car's own removed). Collision
avoidance stays active underneath as the safety net.
"""

import math


class FollowController:
    def __init__(self, params, speed_model, min_gap=0.20, headway=0.9, kp=0.9, lane_half_width=0.13,
                 max_range=2.2):
        self.p = params
        self.model = speed_model
        self.min_gap = min_gap
        self.headway = headway
        self.kp = kp
        self.lane = lane_half_width
        self.max_range = max_range
        self.enabled = False
        self.lead = None            # (gap, lead_speed) while following
        self._v_lead = 0.0

    def find_lead(self, tracks):
        best = None
        for tr in tracks:
            x, y = tr.pos
            if abs(y) <= self.lane + tr.radius and 0.1 < x - self.p.front_x < self.max_range:
                gap = x - self.p.front_x
                if best is None or gap < best[0]:
                    best = (gap, tr)
        return best

    def limit(self, tracks, v_ego, driver_pwm):
        """Returns the PWM the driver's throttle should be capped to (or None if not following)."""
        self.lead = None
        if not self.enabled or driver_pwm <= 0:
            return None
        found = self.find_lead(tracks)
        if found is None:
            self._v_lead *= 0.9
            return None
        gap, tr = found
        v_lead_meas = max(0.0, tr.vel_obj[0]) if tr.moving else 0.0
        self._v_lead += 0.25 * (v_lead_meas - self._v_lead)
        self.lead = (gap, self._v_lead)

        desired = self.min_gap + self.headway * max(v_ego, 0.0)
        v_cmd = max(0.0, self._v_lead + self.kp * (gap - desired))
        if gap < 0.6 * desired:
            v_cmd = 0.0
        return self.model.pwm_for_speed(v_cmd)
