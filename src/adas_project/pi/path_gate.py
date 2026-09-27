"""Path-predicted emergency braking for the relay (replaces the fixed front/rear cones and the +-90 deg body alarm).

The car's real body (20 x 44 cm, from the LiDAR measurements, plus BODY_MARGIN_M on every side) is swept along
the path the driver is commanding right now - and along a slightly tighter and a slightly wider arc, to cover
servo lag and steering slop. Only what that swept body would actually touch counts:
  * passing beside a wall or a box: not in the swept path -> no stop
  * turning into it: in the swept path -> the throttle is limited, then braking
Instead of on/off cancelling (which made the car stutter), the allowed speed falls smoothly with the free
distance:   BASE_M + v * REACTION_S + v^2 / (2 * DECEL)  <=  free distance / FOS
Active (reverse-pulse) braking only if the car is going clearly faster than that.

Points the LiDAR can no longer see (closer than 0.20 m to the sensor) come from an obstacle memory: points
seen a moment ago, moved along with the car's own motion, kept until the car has driven 1.5 m past them.
"""
import math

import numpy as np

from adas.geometry import travel_distance_to_contact
from adas.memory import ObstacleMemory

FOS = 1.3                  # the stopping distance must fit into the free distance 1.3 times over
BODY_MARGIN_M = 0.03       # extra clearance around the whole body
BASE_M = 0.05              # standoff kept at walking pace (bumper to obstacle)
REACTION_S = 0.20          # LiDAR scan + relay + motor delay (fitted command delay 0.12 s + scan period)
DECEL = 1.2                # m/s^2 the car stops at when the throttle is cut (logs: 1-7 cm roll-out at ~0.3 m/s)
KAPPA_SLOP = 0.35          # 1/m: also sweep this much tighter and wider than commanded
HORIZON_M = 1.6
CREEP_V = 0.10             # m/s allowed while there is still more than CREEP_MIN_M free (parking, nosing up)
CREEP_MIN_M = 0.06
BRAKE_OVER_V = 0.15        # brake actively only when this much faster than allowed
BRAKE_GAIN = 300.0         # PWM per m/s over
BRAKE_PWM_MAX = 140


class PathGate:
    def __init__(self, params, speed_model):
        self.p, self.model = params, speed_model
        self.memory = ObstacleMemory(params, blind_radius=0.27, keep_radius=1.2, max_age=60.0, max_points=700,
                                     max_travel=1.5)
        self.pts = np.empty((0, 2))
        self.seq = None
        self.info = {}

    def on_scan(self, pts_vehicle, seq):
        """New LiDAR scan (vehicle frame: x forward from the rear axle, y left)."""
        if seq == self.seq:
            return
        self.seq = seq
        self.pts = pts_vehicle
        self.memory.prune_contradicted(pts_vehicle)     # never trust memory over what the LiDAR sees now
        self.memory.add_scan(pts_vehicle)

    def free_distance(self, delta, direction):
        """How far the rear axle can travel along the commanded path (and its tighter/wider neighbours)
        before the body plus margin touches anything."""
        blind = self.memory.blind_points()
        pts = np.vstack([self.pts, blind]) if len(blind) else self.pts
        if len(pts) == 0:
            return math.inf, 0
        k = math.tan(delta) / self.p.wheelbase
        best = math.inf
        for kk in (k, k + KAPPA_SLOP, k - KAPPA_SLOP):
            d = travel_distance_to_contact(pts, math.atan(kk * self.p.wheelbase), direction, self.p,
                                           horizon=HORIZON_M, margin=BODY_MARGIN_M)
            best = min(best, d)
        return best, len(blind)

    @staticmethod
    def allowed_speed(free):
        """Largest v with FOS * (BASE + v*REACTION + v^2/2a) <= free."""
        room = free / FOS - BASE_M
        if room <= 0:
            return 0.0
        a, r = DECEL, REACTION_S
        return -a * r + math.sqrt((a * r) ** 2 + 2 * a * room)

    def decide(self, dt, physical, delta, v_est, closing=0.0):
        """physical: the throttle to be sent (+ forward). delta: commanded steering angle (rad). v_est: speed
        estimate (+ forward), closing: LiDAR-measured closing speed toward what is ahead in the travel direction.
        Returns (physical to send, braking?)."""
        self.memory.advance(dt, v_est, delta)
        self.info = {}
        if physical == 0:
            return 0, False
        direction = 1 if physical > 0 else -1
        free, n_blind = self.free_distance(delta, direction)
        v = max(abs(v_est) if v_est * direction > 0 else 0.0, closing)
        v_ok = self.allowed_speed(free)
        if free > CREEP_MIN_M:
            v_ok = max(v_ok, CREEP_V)
        else:
            v_ok = 0.0
        self.info = {"free_m": round(free, 3) if math.isfinite(free) else None, "v_allowed": round(v_ok, 2),
                     "blind_pts": n_blind}
        if v > v_ok + BRAKE_OVER_V and free < HORIZON_M:
            pulse = min(BRAKE_PWM_MAX, BRAKE_GAIN * (v - v_ok))
            self.info["action"] = "braking"
            return -direction * int(pulse), True
        cap = self.model.pwm_for_speed(v_ok) if v_ok > 0 else 0.0
        if abs(physical) > cap:
            self.info["action"] = "limited" if cap > 0 else "stopped"
            return direction * int(cap), False
        return physical, False
