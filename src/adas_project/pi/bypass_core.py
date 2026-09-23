"""Obstacle bypass: drive straight, go around whatever is in the path on the side with the
MOST free room, run alongside it, then rejoin the ORIGINAL straight line and carry on.

Pure numpy (no hardware imports) so it can be exercised in a simulator first.

World frame = the LiDAR frame at the moment the run starts: x along the intended path, y toward
positive bearing. `pose` = (x, y, th) of the LiDAR in that frame (from scanmatch.Odometry).

States: CRUISE -> AVOID (arc out) -> PASS (alongside, holding the offset) -> RETURN (arc back
onto y=0) -> DONE.   ABORT if there is no gap wide enough on either side or anything gets
inside the car's footprint margin."""
import math

import numpy as np

from pi.scanmatch import rot

SERVO_STRAIGHT = 86.9        # measured (pi/center_fine.py)
K_CURV_PER_DEG = 0.068       # rad/m of path curvature per servo degree (measured, turn_test/center_fine)
MAX_SERVO_DEG = 24.0

HALF_PATH = 0.17             # obstacle counts as "in the path" if within this of the line (body half-width 0.07 + margin)
LOOK_M = 1.15                # how far ahead an obstacle is noticed
CLEAR_M = 0.20               # side clearance kept between the obstacle edge and the car centerline (0.07 body + margin)
MIN_DETECT_DIST_M = 0.75     # closer than this and the sidestep can't finish before reaching it -> refuse
MIN_GAP_M = 0.45             # a side is only usable if the free gap beside the obstacle is at least this
L_H = 0.45                   # heading-law length constant (m): how quickly y is pulled back to y_ref
TAU_S = 0.20                 # distance over which the heading error is closed (m)
TH_DES_MAX_DEG = 28.0        # never aim more steeply than this
REAR_EXTENT = 0.17           # LiDAR to rear bumper
FRONT_EXTENT = 0.16          # LiDAR to front bumper
BODY_HALF_W = 0.07
STOP_MARGIN = 0.05           # footprint + this = emergency stop zone
EXTRA_STRAIGHT_M = 0.35      # keep going straight this far after rejoining the line


def _cluster(pts, seeds, radius=0.16, max_iter=60):
    """Region-grow: every point within `radius` of the growing set (seeded by `seeds`)."""
    if len(seeds) == 0 or len(pts) == 0:
        return np.zeros(len(pts), bool)
    inside = np.zeros(len(pts), bool)
    frontier = seeds
    for _ in range(max_iter):
        d = np.sqrt(((pts[:, None, :] - frontier[None, :, :]) ** 2).sum(-1)).min(1)
        new = (d < radius) & ~inside
        if not new.any():
            break
        inside |= new
        frontier = pts[new]
    return inside


class Bypass:
    def __init__(self, pwm=115):
        self.pwm = pwm
        self.state = "CRUISE"
        self.side = 0                # +1 / -1 = which way we go around
        self.obs = None              # (xmin, xmax, ymin, ymax) latched obstacle extents
        self.y_target = 0.0
        self.msg = "cruising straight"
        self.rejoined_x = None
        self.gap_pos = self.gap_neg = None

    # --- geometry helpers -------------------------------------------------------------
    @staticmethod
    def to_world(xy, pose):
        return xy @ rot(pose[2]).T + pose[:2]

    def _footprint_violation(self, xy_local):
        """Anything inside the body rectangle + STOP_MARGIN (car frame, LiDAR origin)."""
        if len(xy_local) == 0:
            return False
        x, y = xy_local[:, 0], xy_local[:, 1]
        m = (x > -REAR_EXTENT - STOP_MARGIN) & (x < FRONT_EXTENT + STOP_MARGIN) & (np.abs(y) < BODY_HALF_W + STOP_MARGIN)
        return bool(m.sum() >= 2)

    def step(self, pose, xy_local):
        """pose (x,y,th); xy_local = current scan points (car frame). Returns dict:
        servo_deg (absolute servo command), pwm (0 = stop), state, msg, y_ref."""
        x, y, th = pose
        pw = self.to_world(xy_local, pose) if len(xy_local) else np.zeros((0, 2))
        out = {"pwm": self.pwm, "state": self.state}

        if self.state in ("DONE", "ABORT"):
            return self._out(0, SERVO_STRAIGHT, 0.0)

        if self._footprint_violation(xy_local):
            self.state, self.msg = "ABORT", "EMERGENCY STOP - something inside the safety zone around the car"
            return self._out(0, SERVO_STRAIGHT, 0.0)

        # ---- obstacle in the path? -------------------------------------------------
        if self.state == "CRUISE":
            path = pw[(pw[:, 0] > x + 0.05) & (pw[:, 0] < x + LOOK_M) & (np.abs(pw[:, 1]) < HALF_PATH)] if len(pw) else pw
            if len(path) >= 4:
                near = pw[(pw[:, 0] > x - 0.4) & (pw[:, 0] < x + LOOK_M + 1.0) & (np.abs(pw[:, 1]) < 1.6)]
                cmask = _cluster(near, path)
                cl = near[cmask]
                xmin, xmax = cl[:, 0].min(), cl[:, 0].max()
                ymin, ymax = cl[:, 1].min(), cl[:, 1].max()
                others = near[~cmask]
                band = others[(others[:, 0] > xmin - 0.25) & (others[:, 0] < xmax + 0.25)] if len(others) else others
                pos = band[band[:, 1] > ymax + 0.03][:, 1] if len(band) else np.array([])
                neg = band[band[:, 1] < ymin - 0.03][:, 1] if len(band) else np.array([])
                self.gap_pos = float(pos.min() - ymax) if len(pos) else 2.0
                self.gap_neg = float(ymin - neg.max()) if len(neg) else 2.0
                if xmin - x < MIN_DETECT_DIST_M:
                    self.state = "ABORT"
                    self.msg = (f"obstacle only {xmin - x:.2f} m ahead - too close to steer around it safely "
                                f"(need {MIN_DETECT_DIST_M} m of run-up); more room needed")
                    return self._out(0, SERVO_STRAIGHT, 0.0)
                if max(self.gap_pos, self.gap_neg) < MIN_GAP_M:
                    self.state = "ABORT"
                    self.msg = (f"obstacle ahead but no gap wide enough (free room: +side {self.gap_pos:.2f} m, "
                                f"-side {self.gap_neg:.2f} m, need {MIN_GAP_M} m) - stopping")
                    return self._out(0, SERVO_STRAIGHT, 0.0)
                self.side = 1 if self.gap_pos >= self.gap_neg else -1
                gap = self.gap_pos if self.side > 0 else self.gap_neg
                edge = ymax if self.side > 0 else ymin
                off = min(CLEAR_M, max(0.12, gap / 2 - BODY_HALF_W))
                self.y_target = edge + self.side * off if gap > 2 * CLEAR_M else edge + self.side * gap / 2
                self.obs = [xmin, xmax, ymin, ymax]
                self.state = "AVOID"
                self.msg = (f"OBSTACLE AHEAD at {xmin - x:.2f} m - free gap +side {self.gap_pos:.2f} m / -side "
                            f"{self.gap_neg:.2f} m -> going around the {'+' if self.side > 0 else '-'} side")

        # ---- keep the obstacle extents current while passing ---------------------
        if self.obs is not None and self.state in ("AVOID", "PASS") and len(pw):
            xmin, xmax, ymin, ymax = self.obs
            seeds = pw[(pw[:, 0] > xmin - 0.1) & (pw[:, 0] < xmax + 0.1) & (pw[:, 1] > ymin - 0.1) & (pw[:, 1] < ymax + 0.1)]
            if len(seeds) >= 3:
                self.obs = [min(xmin, seeds[:, 0].min()), max(xmax, seeds[:, 0].max()),
                            min(ymin, seeds[:, 1].min()), max(ymax, seeds[:, 1].max())]

        # ---- state transitions ---------------------------------------------------
        y_ref = 0.0
        if self.state == "AVOID":
            y_ref = self.y_target
            if abs(y - self.y_target) < 0.07 and abs(th) < math.radians(10):
                self.state, self.msg = "PASS", "alongside the obstacle - holding the offset"
        if self.state == "PASS":
            y_ref = self.y_target
            if x - REAR_EXTENT > self.obs[1] + 0.06:
                self.state, self.msg = "RETURN", "obstacle passed - returning to the original line"
        if self.state == "RETURN":
            y_ref = 0.0
            if abs(y) < 0.05 and abs(th) < math.radians(3):
                if self.rejoined_x is None:
                    self.rejoined_x = x
                    self.msg = "back on the original line - continuing straight"
                if x - self.rejoined_x > EXTRA_STRAIGHT_M:
                    self.state, self.msg = "DONE", "DONE - rejoined the original straight path"
                    return self._out(0, SERVO_STRAIGHT, 0.0)

        # ---- heading-tracking steering (damped: no overshoot when rejoining the line) -------
        # desired heading points back toward y_ref with a length constant L_H; the path curvature
        # then closes the heading error over a distance TAU_S.  Second-order response (zeta ~0.66).
        y_err = y - y_ref
        th_des = -math.atan(y_err / L_H)
        th_des = max(-math.radians(TH_DES_MAX_DEG), min(math.radians(TH_DES_MAX_DEG), th_des))
        kappa = (th_des - th) / TAU_S
        servo_off = max(-MAX_SERVO_DEG, min(MAX_SERVO_DEG, kappa / K_CURV_PER_DEG))
        return self._out(self.pwm, SERVO_STRAIGHT + servo_off, y_ref, kappa)

    def _out(self, pwm, servo, y_ref, kappa=0.0):
        return {"pwm": pwm, "servo_deg": servo, "state": self.state, "msg": self.msg,
                "y_ref": y_ref, "kappa": kappa, "side": self.side, "obs": self.obs,
                "gap_pos": self.gap_pos, "gap_neg": self.gap_neg}
