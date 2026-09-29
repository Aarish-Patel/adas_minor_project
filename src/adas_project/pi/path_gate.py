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
# Driver trusted by the learned intent model AND active on the stick: the soft speed cap may wait until closer to
# the last point to brake (threat assessment as in Brannstrom, Coelingh & Sjoberg, IEEE T-ITS 2010). Tested in the
# Monte Carlo at 1.1 and 1.0: needless speed limits fell but became needless brakes and swerves (21 -> 22 in total),
# so it is OFF (= FOS). Active braking stays at the physical limit for every driver either way.
FOS_TRUSTED = FOS
# ...instead, a trusted driver who is steering is judged on their predicted (still-curving) path, but the soft cap
# never goes above this safety factor on the frozen-steering path - so it can still slow the car smoothly before the
# physical limit would need a hard brake (going all the way to 1.0 turned limits into brakes for fast drivers)
TRUSTED_FOS_FLOOR = 1.15
BODY_MARGIN_M = 0.03       # extra clearance around the whole body (at speed)
# Speed-dependent protective field, as laser scanners on AGVs switch field size with speed (ISO 3691-4): up to each
# speed the body margin is smaller, so the car can slow down and fit through a tight gap it would be refused at
# speed. (speed limit m/s, margin m); above the last level BODY_MARGIN_M applies.
MARGIN_LEVELS = ((0.15, 0.012), (0.40, 0.02))
BASE_M = 0.05              # standoff kept at walking pace (bumper to obstacle)
REACTION_S = 0.20          # LiDAR scan + relay + motor delay (fitted command delay 0.12 s + scan period)
DECEL = 4.0                # m/s^2 the car stops at when the throttle is cut, after the delay in REACTION_S: relay
                           # drive log 28 Sep - ~6 cm roll-out from 0.7 m/s on a cut, 1-3 cm with an active brake
                           # pulse (user: "brakes almost instantly"). Was 1.2 (too pessimistic: early interventions)
KAPPA_SLOP = 0.35          # 1/m: the old fixed band (kept for comparison: GATE_BAND = "fixed")
# Steering the car can actually be on before the next decision (least-restrictive safety filter, Hsu/Hu/Fisac 2023):
# every arc between the previous command and this one (servo slew during the command delay), widened by the
# steering model's error - 0.08 1/m plus 15 % of the curvature (the fitted left/right gains differ by ~+-15 %).
K_WINDOW_S = 0.25          # s the servo may still follow an earlier command (fitted delay 0.12 s + slew + a tick)
KAPPA_ERR_ABS = 0.08
KAPPA_ERR_REL = 0.15
GATE_BAND = "transition"
HORIZON_M = 1.6
CREEP_V = 0.10             # m/s allowed while there is still more than CREEP_MIN_M free (parking, nosing up)
CREEP_MIN_M = 0.06
BRAKE_OVER_V = 0.15        # brake actively only when this much faster than allowed
BRAKE_GAIN = 300.0         # PWM per m/s over
BRAKE_PWM_MAX = 80         # peak reverse pulse (was 140: the car stops within 1-2 cm on a throttle cut alone, and
                           # reverse pulses at speed shock-load the drivetrain - the rear shaft broke on 28 Sep)
BRAKE_MAX_S = 0.2          # s of active braking per event (never enough to drive the car backwards)
LATCH_RELEASE_M = 0.08     # the hold after braking lets go once the free distance grows this much
# ...or, once the brake pulse is over and the car stands still, when the commanded path is clear at SOME speed (user,
# 28 Sep: a manoeuvre steering round the obstacle must not stay held because the old path was blocked - go at the
# speed the new path allows; the gate keeps checking it every tick)
# steering correction instead of braking (steer_correction): the largest nudge of path curvature tried (0.7 1/m is
# ~11 servo degrees at the fitted 0.0656 1/m per degree), tried smallest first, and only above walking pace
NUDGE_MAX_KAPPA = 0.7
NUDGE_STEPS = (0.1, 0.2, 0.3, 0.45, 0.6, 0.7)
NUDGE_MIN_V = 0.2
NUDGE_TTC_S = 1.0          # s: look this far ahead in time on the driver's path for a contact to correct
NUDGE_GAIN_M = 0.5         # m: a correction must add at least this much free way (i.e. get past the obstacle)


class PathGate:
    def __init__(self, params, speed_model):
        self.p, self.model = params, speed_model
        self.delay_source = None        # an object with .excess (s): extra scan latency measured online (adas/latency.py)
        self.memory = ObstacleMemory(params, blind_radius=0.27, keep_radius=1.2, max_age=60.0, max_points=700,
                                     max_travel=1.5)
        self.pts = np.empty((0, 2))
        self.seq = None
        self.info = {}
        self.latch = None          # (direction, free distance) while holding after a brake
        self.brake_t = 0.0
        self.k_hist = []           # (age s, curvature) of recent steering commands: where the servo may still be
        self._v = 0.0              # speed estimate of the current decision (sizes the delay segment)

    def on_scan(self, pts_vehicle, seq):
        """New LiDAR scan (vehicle frame: x forward from the rear axle, y left)."""
        if seq == self.seq:
            return
        self.seq = seq
        self.pts = pts_vehicle
        self.memory.prune_contradicted(pts_vehicle)     # never trust memory over what the LiDAR sees now
        self.memory.add_scan(pts_vehicle)

    def free_distance(self, delta, direction, slop=True, margin=BODY_MARGIN_M):
        """How far the rear axle can travel along the commanded path (and its tighter/wider neighbours)
        before the body plus margin touches anything."""
        blind = self.memory.blind_points()
        pts = np.vstack([self.pts, blind]) if len(blind) else self.pts
        if len(pts) == 0:
            return math.inf, 0
        pts = self._drop_receding(pts, delta, direction, margin=margin)
        if len(pts) == 0:
            return math.inf, len(blind)
        k = math.tan(delta) / self.p.wheelbase
        if not slop:
            return travel_distance_to_contact(pts, delta, direction, self.p, horizon=HORIZON_M, margin=margin), len(blind)
        if GATE_BAND == "fixed":
            best = math.inf
            for kk in (k, k + KAPPA_SLOP, k - KAPPA_SLOP):
                best = min(best, travel_distance_to_contact(pts, math.atan(kk * self.p.wheelbase), direction, self.p,
                                                            horizon=HORIZON_M, margin=margin))
            return best, len(blind)
        # the path the car can actually take before the next decision: the servo may still be on any command of the
        # last K_WINDOW_S for that long (command delay + slew), then it follows this command - each widened by the
        # steering model's error
        d_pre = max(abs(self._v), 0.10) * K_WINDOW_S
        hist = [kk for _, kk in self.k_hist] or [k]
        pres = sorted({self._widen(min(hist), -1), self._widen(max(hist), +1)})
        posts = (self._widen(k, -1), k, self._widen(k, +1))
        best = math.inf
        for kp in pres:
            for kq in posts:
                best = min(best, self._composite_contact(pts, kp, d_pre, kq, direction, margin))
        return best, len(blind)

    PATH_TRACK_TOL_M = 0.02        # extra margin for how far the car may be off a planned path while following it

    def free_along_leg(self, leg, margin=BODY_MARGIN_M):
        """Free travel along a PLANNED path leg (poses (N, 3) in the vehicle frame, the car at its start): the
        distance to the first pose where the body plus margin (and a path-tracking tolerance) touches anything;
        HORIZON_M if the whole leg is clear (the car follows the leg, so the arc it is on right now does not count)."""
        blind = self.memory.blind_points()
        pts = np.vstack([self.pts, blind]) if len(blind) else self.pts
        if len(pts) == 0 or len(leg) == 0:
            return HORIZON_M
        m = margin + self.PATH_TRACK_TOL_M
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(leg[:, 0]), np.diff(leg[:, 1])))])
        near = pts[np.hypot(pts[:, 0], pts[:, 1]) < s[-1] + self.p.front_x + 0.5]
        if len(near) == 0:
            return HORIZON_M
        c, sn = np.cos(leg[:, 2])[:, None], np.sin(leg[:, 2])[:, None]
        dx, dy = near[None, :, 0] - leg[:, 0][:, None], near[None, :, 1] - leg[:, 1][:, None]
        lx, ly = c * dx + sn * dy, -sn * dx + c * dy
        hit = ((lx >= self.p.rear_x - m) & (lx <= self.p.front_x + m) & (np.abs(ly) <= self.p.width / 2 + m)).any(axis=1)
        idx = np.flatnonzero(hit)
        return float(s[idx[0]]) if len(idx) else HORIZON_M

    @staticmethod
    def _widen(k, side):
        return k + side * (KAPPA_ERR_ABS + KAPPA_ERR_REL * abs(k))

    def _composite_contact(self, pts, k_pre, d_pre, k_post, direction, margin):
        """Travel to contact along k_pre for d_pre metres, then along k_post."""
        wb = self.p.wheelbase
        d1 = travel_distance_to_contact(pts, math.atan(k_pre * wb), direction, self.p, horizon=d_pre, margin=margin)
        if d1 <= d_pre:
            return d1
        s = direction * d_pre
        th = k_pre * s
        x, y = (s, 0.0) if abs(k_pre) < 1e-6 else (math.sin(th) / k_pre, (1 - math.cos(th)) / k_pre)
        c, sn = math.cos(th), math.sin(th)
        dx, dy = pts[:, 0] - x, pts[:, 1] - y
        local = np.column_stack([c * dx + sn * dy, -sn * dx + c * dy])
        return d_pre + travel_distance_to_contact(local, math.atan(k_post * wb), direction, self.p,
                                                  horizon=HORIZON_M - d_pre, margin=margin)

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

    def _drop_receding(self, pts, delta, direction, probe=0.05, margin=BODY_MARGIN_M):
        """Points already inside the body margin right now would make every direction look blocked (the car
        freezes next to a box, unable even to back away). Such points only count if the next few centimetres of
        this motion bring the body closer to them."""
        d0 = self._body_dist(pts, 0.0, 0.0, 0.0)
        m = margin + 0.01                     # the same square-cornered margin box the path sweep uses
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

    def steer_correction(self, delta, direction, v, max_shift=None):
        """Minimal-intervention steering (TODO B8/L2; Hsu, Hu & Fisac 2024 safety filters; Talbot et al., ACC 2025
        shared-control CBFs): if the brake would have to act on the driver's path at this speed, the smallest change
        of path curvature (up to max_shift, 1/m) whose path IS safe at this speed - like the emergency steering
        support in production cars. Returns the corrected steering angle, or None (no correction needed, or none
        small enough: then the brake acts as before)."""
        from adas.vehicle_params import steer_to_delta
        max_shift = NUDGE_MAX_KAPPA if max_shift is None else max_shift
        if direction == 0 or abs(v) < NUDGE_MIN_V:
            return None
        v = abs(v)
        free0, _ = self.free_distance(delta, direction)
        # act while a small correction can still get round (the brake's envelope is too late for steering: at
        # 0.5 m/s it is ~0.2 m, a 6 cm sideways slide needs ~0.4 m) - a time-to-contact trigger as in production
        # evasive steering support, or the brake envelope if that is larger
        envelope = FOS * (BASE_M + v * self.reaction_s() + v * v / (2 * DECEL))
        if not math.isfinite(free0) or free0 > max(v * NUDGE_TTC_S, envelope):
            return None
        need = max(free0 + NUDGE_GAIN_M, envelope)        # the corrected path must really get past it
        wb = self.p.wheelbase
        k0 = math.tan(delta) / wb
        k_lo, k_hi = math.tan(steer_to_delta(-1.0, self.p)) / wb, math.tan(steer_to_delta(1.0, self.p)) / wb
        for shift in NUDGE_STEPS:
            if shift > max_shift:
                break
            for k in (k0 + shift, k0 - shift):
                if not k_lo <= k <= k_hi:
                    continue
                free_k, _ = self.free_distance(math.atan(k * wb), direction)
                if free_k >= need and self.allowed_speed(free_k) >= v:
                    self.info["nudge"] = f"steering corrected {'left' if k > k0 else 'right'} by {shift:.2f} 1/m"
                    return math.atan(k * wb)
        return None

    def reaction_s(self):
        """The delay the stopping distance budgets: the design value plus the extra scan latency measured online."""
        return REACTION_S + (self.delay_source.excess if self.delay_source is not None else 0.0)

    def allowed_speed(self, free, fos=FOS):
        """Largest v with fos * (BASE + v*REACTION + v^2/2a) <= free."""
        room = free / fos - BASE_M
        if room <= 0:
            return 0.0
        a, r = DECEL, self.reaction_s()
        return -a * r + math.sqrt((a * r) ** 2 + 2 * a * room)

    def decide(self, dt, physical, delta, v_est, closing=0.0, intent_k_rate=None, trusted=False, leg=None):
        """physical: the throttle to be sent (+ forward). delta: commanded steering angle (rad). v_est: speed
        estimate (+ forward), closing: LiDAR-measured closing speed toward what is ahead in the travel direction.
        trusted: the learned intent model expects this (attentive) driver to handle it - later soft cap only.
        leg: (poses (N, 3) vehicle frame, direction) of a planned path the car is following (evasive manoeuvre or
        click-to-go) - then the free distance is measured ALONG that path: a path that is clear at a slower speed is
        driven at that speed instead of the car being held because its current arc points at the obstacle.
        Returns (physical to send, braking?)."""
        fos = FOS_TRUSTED if trusted else FOS
        self.memory.advance(dt, v_est, delta)
        self.info = {}
        self.k_hist = [(age + dt, kk) for age, kk in self.k_hist if age + dt <= K_WINDOW_S]
        k_now = math.tan(delta) / self.p.wheelbase
        self._v = v_est
        if physical == 0:
            self.latch = None                     # the driver let go: the hold is released
            self.k_hist.append((0.0, k_now))
            return 0, False
        direction = 1 if physical > 0 else -1
        free, n_blind = self.free_distance(delta, direction)      # the steering the car can physically be on
        self.k_hist.append((0.0, k_now))
        # a planned path counts only while the car is really on it (within 5 cm / 10 deg of its start), and the arc
        # the car is on right now still counts for the distance covered before the steering can change
        # (the path point nearest the car counts - a coarse search's first point is up to ~6 cm ahead of it - and
        # the sweep starts at the car itself)
        on_leg = False
        if leg is not None and leg[1] == direction and len(leg[0]) >= 2:
            L = leg[0]
            j = int(np.argmin(np.hypot(L[:, 0], L[:, 1])))
            if math.hypot(L[j, 0], L[j, 1]) < 0.08 and abs(math.remainder(L[j, 2], 2 * math.pi)) < math.radians(12):
                leg = (np.vstack([[0.0, 0.0, 0.0], L[j:]]), leg[1])
                on_leg = True
        d_pre = max(abs(v_est), 0.10) * K_WINDOW_S + 0.05
        if on_leg:
            f_leg = self.free_along_leg(leg[0])
            free = f_leg if free > d_pre else min(free, f_leg)
            self.info["leg"] = True
        v = max(abs(v_est) if v_est * direction > 0 else 0.0, closing)
        v_ok = self.allowed_speed(free, fos)
        v_phys = self.allowed_speed(free, 1.0)     # the physical limit: brake above this whoever is driving
        f_creep = free
        if v_ok < MARGIN_LEVELS[-1][0]:            # slower speeds may use their smaller protective field
            for v_lvl, m in MARGIN_LEVELS[::-1]:
                f_arc = self.free_distance(delta, direction, margin=m)[0]
                f_creep = f_arc
                if on_leg:
                    f_l = self.free_along_leg(leg[0], margin=m)
                    f_creep = f_l if f_arc > d_pre else min(f_arc, f_l)
                v_l = min(v_lvl, self.allowed_speed(f_creep, fos))
                if v_l > v_ok:
                    v_ok, free = v_l, f_creep
                    v_phys = max(v_phys, min(v_lvl, self.allowed_speed(f_creep, 1.0)))
        if f_creep > CREEP_MIN_M:
            v_ok = max(v_ok, CREEP_V)
            free = max(free, f_creep)
        else:
            v_ok = 0.0
        # intent-aware soft cap: a driver the learned model trusts AND who is steering right now (intent_k_rate =
        # their curvature rate) is predicted along their still-curving path, so turning away from an obstacle is not
        # slowed. The frozen-steering path keeps its physical limit (v_phys): if the driver stops steering, the brake
        # below still stops the car in time. Close in, only the physical path counts.
        if trusted and intent_k_rate is not None and free > INTENT_FLOOR_M:
            blind = self.memory.blind_points()
            pts = np.vstack([self.pts, blind]) if len(blind) else self.pts
            pts = self._drop_receding(pts, delta, direction)
            free_i = contact_along_changing_curvature(pts, k_now, intent_k_rate, v_est, direction, self.p,
                                                      horizon=HORIZON_M, margin=BODY_MARGIN_M)
            free_i = min(free_i, free + INTENT_MAX_GAIN_M)
            v_i = min(self.allowed_speed(free_i, FOS), max(self.allowed_speed(free, TRUSTED_FOS_FLOOR), v_ok))
            if v_i > v_ok:
                self.info["intent"] = f"driver is steering away - predicted path free {min(free_i, 9):.2f} m"
                v_ok = v_i
        self.info.update({"free_m": round(free, 3) if math.isfinite(free) else None, "v_allowed": round(v_ok, 2),
                          "blind_pts": n_blind})
        # active braking: clearly over the soft cap - and for a trusted driver (whose soft cap can be later) no
        # later than the physical limit of the frozen-steering path
        v_brake = v_ok + BRAKE_OVER_V
        if trusted:
            v_brake = min(v_brake, max(v_phys, v_ok) + 0.03)
            self.info["trusted"] = True
        # latch: after braking for an obstacle, hold the throttle at zero toward it (no brake/throttle/brake
        # stutter) until the driver lets go or reverses, or the free distance has clearly grown again
        if self.latch is not None:
            l_dir, l_free, l_delta = self.latch
            # released onto a path that is clear at a slower speed: the steering turned to a new one, or the car has
            # stopped and the commanded path is clear at a creep (the cap below keeps it within its envelope, so
            # there is no brake/throttle stutter to prevent any more)
            # never in the middle of a brake pulse (released, braked again next tick: brake/release hammering
            # that shock-loads the drivetrain - seen in the 28 Sep drive log before the rear shaft broke)
            settled = self.brake_t >= BRAKE_MAX_S and abs(v_est) < 0.05
            new_way = v_ok > 0 and settled
            if l_dir != direction or free > l_free + LATCH_RELEASE_M or new_way:
                self.latch = None
                if new_way:
                    self.info["released"] = "the new steering is clear at a slower speed - going on slowly"
            else:
                self.brake_t += dt
                if self.brake_t < BRAKE_MAX_S and v > v_brake:
                    self.info["action"] = "braking"
                    return -direction * int(min(BRAKE_PWM_MAX, BRAKE_GAIN * (v - v_ok))), True
                self.info["action"] = "holding (stopped for an obstacle - release the throttle)"
                return 0, False
        if v > v_brake and free < HORIZON_M:
            self.latch = (direction, free, delta)
            self.brake_t = 0.0
            pulse = min(BRAKE_PWM_MAX, BRAKE_GAIN * (v - v_ok))
            self.info["action"] = "braking"
            return -direction * int(pulse), True
        cap = self.model.pwm_for_speed(v_ok) if v_ok > 0 else 0.0
        if abs(physical) > cap:
            self.info["action"] = "limited" if cap > 0 else "stopped"
            return direction * int(cap), False
        return physical, False
