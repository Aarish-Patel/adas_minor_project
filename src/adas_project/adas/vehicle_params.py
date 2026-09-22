"""Car dimensions and the steering command -> bicycle-model angle mapping.

Frame: metres, x forward, y left, origin at the centre of the rear axle.
Steering command s runs -1 (full right) .. +1 (full left), the same value the
controller sends (steering percent / 100).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleParams:
    wheelbase: float = 0.20        # front axle centre to rear axle centre
    pivot_track: float = 0.07      # distance between the two steering pivots
    width: float = 0.14            # outside of tyre to outside of tyre
    front_overhang: float = 0.06   # body ahead of the front axle   (MEASURE)
    rear_overhang: float = 0.05    # body behind the rear axle      (MEASURE)

    lidar_x: float = 0.12          # LiDAR position ahead of the rear axle (MEASURE)
    lidar_y: float = 0.0

    # Inner-wheel angle at full stick (matches the servo travel in rc_controller.py)
    max_inner_left_deg: float = 50.0
    max_inner_right_deg: float = 40.0

    @property
    def front_x(self):
        return self.wheelbase + self.front_overhang

    @property
    def rear_x(self):
        return -self.rear_overhang


def steer_to_delta(s, p):
    """Steering command -> equivalent single-track (bicycle) steering angle in radians.

    Mirrors rc_controller.steering_to_servo_angles: the inner wheel angle is
    proportional to the stick and the outer wheel follows the Ackermann rule
    cot(outer) - cot(inner) = track / wheelbase. The bicycle angle is the one
    whose cotangent is the mean of the two: cot(delta) = cot(inner) + track/(2*wheelbase).
    """
    s = max(-1.0, min(1.0, s))
    if abs(s) < 1e-9:
        return 0.0
    max_inner = p.max_inner_left_deg if s > 0 else p.max_inner_right_deg
    inner = math.radians(abs(s) * max_inner)
    cot_delta = 1.0 / math.tan(inner) + p.pivot_track / (2.0 * p.wheelbase)
    return math.copysign(math.atan(1.0 / cot_delta), s)


def steer_to_wheel_angles(s, p):
    """Steering command -> (left wheel, right wheel) angles in radians, + = left. Ackermann."""
    s = max(-1.0, min(1.0, s))
    if abs(s) < 1e-9:
        return 0.0, 0.0
    max_inner = p.max_inner_left_deg if s > 0 else p.max_inner_right_deg
    inner = math.radians(abs(s) * max_inner)
    outer = math.atan(1.0 / (1.0 / math.tan(inner) + p.pivot_track / p.wheelbase))
    if s > 0:
        return inner, outer
    return -outer, -inner


_STEER_TABLE = {}


def delta_to_steer(delta, p):
    """Inverse of steer_to_delta: the stick value that produces a given bicycle steering angle."""
    import numpy as np
    key = (p.wheelbase, p.pivot_track, p.max_inner_left_deg, p.max_inner_right_deg)
    if key not in _STEER_TABLE:
        s = np.linspace(-1.0, 1.0, 401)
        _STEER_TABLE[key] = (np.array([steer_to_delta(x, p) for x in s]), s)
    d, s = _STEER_TABLE[key]
    return float(np.interp(delta, d, s))
