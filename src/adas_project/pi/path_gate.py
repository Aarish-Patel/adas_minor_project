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

from adas.geometry import contact_along_changing_curvature, travel_distance_to_contact
from adas.memory import ObstacleMemory

INTENT_MIN_DELTA = 0.03    # rad: the steering trend must be meaningful
INTENT_FLOOR_M = 0.12      # m: below this free distance only the current path counts
INTENT_MAX_GAIN_M = 1.2    # m: the prediction can add at most this much free distance
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
BRAKE_MAX_S = 0.3          # s of active braking per event (never enough to drive the car backwards)
LATCH_RELEASE_M = 0.08     # the hold after braking lets go once the free distance grows this much


class PathGate:
    def __init__(self, params, speed_model):
        self.p, self.model = params, speed_model
        self.memory = ObstacleMemory(params, blind_radius=0.27, keep_radius=1.2, max_age=60.0, max_points=700,
                                     max_travel=1.5)
        self.pts = np.empty((0, 2))
        self.seq = None
        self.info = {}
        self.latch = None          # (direction, free distance) while holding after a brake
        self.brake_t = 0.0

    def on_scan(self, pts_vehicle, seq):
        """New LiDAR scan (vehicle frame: x forward from the rear axle, y left)."""
        if seq == self.seq:
            return
        self.seq = seq
        self.pts = pts_vehicle
        self.memory.prune_contradicted(pts_vehicle)     # never trust memory over what the LiDAR sees now
        self.memory.add_scan(pts_vehicle)

    def free_distance(self, delta, direction, slop=True):
        """How far the rear axle can travel along the commanded path (and its tighter/wider neighbours)
        before the body plus margin touches anything."""
        blind = self.memory.blind_points()
        pts = np.vstack([self.pts, blind]) if len(blind) else self.pts
        if len(pts) == 0:
            return math.inf, 0
        pts = self._drop_receding(pts, delta, direction)
        if len(pts) == 0:
            return math.inf, len(blind)
        k = math.tan(delta) / self.p.wheelbase
        best = math.inf
        for kk in ((k, k + KAPPA_SLOP, k - KAPPA_SLOP) if slop else (k,)):
            d = travel_distance_to_contact(pts, math.atan(kk * self.p.wheelbase), direction, self.p,
                                           horizon=HORIZON_M, margin=BODY_MARGIN_M)
            best = min(best, d)
        return best, len(blind)

    def overlay(self, delta, direction, v, v_cmd, horizon_s=2.5):
        """What the GUI draws: the path the body will follow at the current stick and throttle (the front of the
        car, and the two front corners = the swept width), and where it would first touch something (X), with
        distance and time to contact. All in the GUI's polar form (angle clockwise from ahead, distance from
        the LiDAR)."""
        if direction == 0:
            return None
        blind = self.memory.blind_points()
        pts = np.vstack([self.pts, blind]) if len(blind) else self.pts
        hit = travel_distance_to_contact(pts, delta, direction, self.p, horizon=HORIZON_M, margin=0.0) \
            if len(pts) else math.inf
        speed = max(abs(v), abs(v_cmd), 0.12)
        length = min(HORIZON_M, speed * horizon_s)
        end = min(length, hit) if math.isfinite(hit) else length
        k = math.tan(delta) / self.p.wheelbase
        s = direction * np.linspace(0.0, max(end, 0.02), 30)
        if abs(k) < 1e-6:
            px, py, psi = s, np.zeros_like(s), np.zeros_like(s)
        else:
            psi = k * s
            px, py = np.sin(psi) / k, (1 - np.cos(psi)) / k
        tip = self.p.front_x if direction > 0 else self.p.rear_x
        hw = self.p.width / 2
        c, sn = np.cos(psi), np.sin(psi)

        def polar(xs, ys):
            dx, dy = xs - self.p.lidar_x, ys - self.p.lidar_y
            return [[round(float(-math.degrees(math.atan2(b, a))), 1), round(float(math.hypot(a, b)), 3)]
                    for a, b in zip(dx, dy)]
        centre = polar(px + c * tip, py + sn * tip)
        left = polar(px + c * tip - sn * hw, py + sn * tip + c * hw)
        right = polar(px + c * tip + sn * hw, py + sn * tip - c * hw)
        out = {"pred": centre, "left": left, "right": right, "hit": None, "hit_m": None, "ttc": None,
               "state": self.info.get("action") and ("limited" if self.info["action"] == "limited" else "collision")
               or "clear"}
        if math.isfinite(hit) and hit <= length:
            out["hit"], out["hit_m"] = centre[-1], round(hit, 2)
            out["ttc"] = round(hit / speed, 1)
            out["state"] = "collision" if hit < max(0.3, speed * 1.5) else (
                "limited" if out["state"] == "clear" else out["state"])
        return out

    def _body_dist(self, pts, x, y, th):
        c, s = math.cos(th), math.sin(th)
        dx, dy = pts[:, 0] - x, pts[:, 1] - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        ex = np.maximum(np.maximum(self.p.rear_x - lx, 0.0), lx - self.p.front_x)
        ey = np.maximum(np.abs(ly) - self.p.width / 2, 0.0)
        return np.hypot(ex, ey)

    def _drop_receding(self, pts, delta, direction, probe=0.05):
        """Points already inside the body margin right now would make every direction look blocked (the car
        freezes next to a box, unable even to back away). Such points only count if the next few centimetres of
        this motion bring the body closer to them."""
        d0 = self._body_dist(pts, 0.0, 0.0, 0.0)
        m = BODY_MARGIN_M + 0.01              # the same square-cornered margin box the path sweep uses
        close = (pts[:, 0] >= self.p.rear_x - m) & (pts[:, 0] <= self.p.front_x + m) & \
                (np.abs(pts[:, 1]) <= self.p.width / 2 + m)
        if not close.any():
            return pts
        k = math.tan(delta) / self.p.wheelbase
        s = direction * probe
        if abs(k) < 1e-6:
            x1, y1, th1 = s, 0.0, 0.0
        else:
            th1 = k * s
            x1, y1 = math.sin(th1) / k, (1 - math.cos(th1)) / k
        d1 = self._body_dist(pts, x1, y1, th1)
        keep = ~close | (d1 < d0 - 1e-4)
        return pts[keep]

    @staticmethod
    def allowed_speed(free):
        """Largest v with FOS * (BASE + v*REACTION + v^2/2a) <= free."""
        room = free / FOS - BASE_M
        if room <= 0:
            return 0.0
        a, r = DECEL, REACTION_S
        return -a * r + math.sqrt((a * r) ** 2 + 2 * a * room)

    def decide(self, dt, physical, delta, v_est, closing=0.0, intent_k_rate=None):
        """physical: the throttle to be sent (+ forward). delta: commanded steering angle (rad). v_est: speed
        estimate (+ forward), closing: LiDAR-measured closing speed toward what is ahead in the travel direction.
        Returns (physical to send, braking?)."""
        self.memory.advance(dt, v_est, delta)
        self.info = {}
        if physical == 0:
            self.latch = None                     # the driver let go: the hold is released
            return 0, False
        direction = 1 if physical > 0 else -1
        free, n_blind = self.free_distance(delta, direction)
        # intent-aware (only for an ATTENTIVE driver - intent_k_rate is None when the stick shows no recent
        # activity, i.e. a lapse): predict the path with the curvature changing at the driver's current steering
        # rate. A driver already steering away from the obstacle is predicted to miss it, so a clip of the frozen
        # current arc is not treated as a threat. Close in, only the current path counts.
        if intent_k_rate is not None and free > INTENT_FLOOR_M:
            blind = self.memory.blind_points()
            pts = np.vstack([self.pts, blind]) if len(blind) else self.pts
            pts = self._drop_receding(pts, delta, direction)
            k0 = math.tan(delta) / self.p.wheelbase
            free_i = contact_along_changing_curvature(pts, k0, intent_k_rate, v_est, direction, self.p,
                                                      horizon=HORIZON_M, margin=BODY_MARGIN_M)
            if free_i > free:
                self.info["intent"] = f"driver is steering away - predicted path free {min(free_i, 9):.2f} m"
                free = min(free_i, free + INTENT_MAX_GAIN_M)
        v = max(abs(v_est) if v_est * direction > 0 else 0.0, closing)
        v_ok = self.allowed_speed(free)
        if free > CREEP_MIN_M:
            v_ok = max(v_ok, CREEP_V)
        else:
            v_ok = 0.0
        self.info.update({"free_m": round(free, 3) if math.isfinite(free) else None, "v_allowed": round(v_ok, 2),
                          "blind_pts": n_blind})
        # latch: after braking for an obstacle, hold the throttle at zero toward it (no brake/throttle/brake
        # stutter) until the driver lets go or reverses, or the free distance has clearly grown again
        if self.latch is not None:
            l_dir, l_free = self.latch
            if l_dir != direction or free > l_free + LATCH_RELEASE_M:
                self.latch = None
            else:
                self.brake_t += dt
                if self.brake_t < BRAKE_MAX_S and v > v_ok + BRAKE_OVER_V:
                    self.info["action"] = "braking"
                    return -direction * int(min(BRAKE_PWM_MAX, BRAKE_GAIN * (v - v_ok))), True
                self.info["action"] = "holding (stopped for an obstacle - release the throttle)"
                return 0, False
        if v > v_ok + BRAKE_OVER_V and free < HORIZON_M:
            self.latch = (direction, free)
            self.brake_t = 0.0
            pulse = min(BRAKE_PWM_MAX, BRAKE_GAIN * (v - v_ok))
            self.info["action"] = "braking"
            return -direction * int(pulse), True
        cap = self.model.pwm_for_speed(v_ok) if v_ok > 0 else 0.0
        if abs(physical) > cap:
            self.info["action"] = "limited" if cap > 0 else "stopped"
            return direction * int(cap), False
        return physical, False
