"""Marker-guided parking (forward into a bay with an ArUco marker on its back wall).

Approach: a Stanley controller follows the bay's centre line (through the marker,
along its facing direction), so the car straightens out on the line and drives
straight in. The marker's position is accurate; its facing angle is noisy from far
away, so it is filtered and weighted by how large the marker appears.

The marker pose is kept in a dead-reckoned frame (from speed and steering), so the
controller keeps working after the marker leaves the camera's view close up.
The final stop is the LiDAR's job (a wall closer than the stopping gap ends the
manoeuvre), and normal collision avoidance stays active underneath.
"""

import math

from .vehicle_params import delta_to_steer, steer_to_delta

IDLE, APPROACH, DONE, FAILED = "idle", "approach", "done", "failed"


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class ParkingController:
    def __init__(self, params, speed_model, marker_id=7, stop_gap=0.11, k_cross=2.4,
                 v_max=0.32, v_min=0.13, lost_timeout=6.0):
        self.p = params
        self.model = speed_model
        self.marker_id = marker_id
        self.stop_gap = stop_gap
        self.k = k_cross
        self.v_max, self.v_min = v_max, v_min
        self.lost_timeout = lost_timeout
        self.state = IDLE
        self.message = "idle"
        self._reset_pose()

    def _reset_pose(self):
        self.ox = self.oy = self.oth = 0.0            # dead-reckoned pose of the car
        self.m = None                                 # filtered marker (x, y, facing) in that frame
        self.last_seen = None
        self.t = 0.0
        self.best_size = 0.0

    @property
    def active(self):
        return self.state == APPROACH

    def start(self):
        self._reset_pose()
        self.state = APPROACH
        self.message = "searching for the marker"

    def stop(self):
        self.state = IDLE
        self.message = "idle"

    def _advance(self, dt, v, delta):
        omega = v * math.tan(delta) / self.p.wheelbase
        mid = self.oth + 0.5 * omega * dt
        self.ox += v * math.cos(mid) * dt
        self.oy += v * math.sin(mid) * dt
        self.oth += omega * dt

    def _to_odom(self, x, y):
        c, s = math.cos(self.oth), math.sin(self.oth)
        return self.ox + c * x - s * y, self.oy + s * x + c * y

    def _to_vehicle(self, X, Y):
        c, s = math.cos(self.oth), math.sin(self.oth)
        dx, dy = X - self.ox, Y - self.oy
        return c * dx + s * dy, -s * dx + c * dy

    def _observe(self, obs_list, t):
        for o in obs_list:
            if o.id != self.marker_id:
                continue
            X, Y = self._to_odom(o.x, o.y)
            facing = wrap(o.facing + self.oth)
            w_pos = 0.6
            w_face = min(0.5, max(0.03, o.size_px / 500.0))
            if self.m is None:
                self.m = [X, Y, facing]
            else:
                self.m[0] += w_pos * (X - self.m[0])
                self.m[1] += w_pos * (Y - self.m[1])
                self.m[2] = wrap(self.m[2] + w_face * wrap(facing - self.m[2]))
            self.last_seen = t
            self.best_size = max(self.best_size, o.size_px)

    def update(self, dt, obs_list, v_est, steer_now, front_gap):
        """Returns (steer, pwm) while active, or None when idle/done/failed.

        front_gap: LiDAR distance ahead to the nearest obstacle in the car's path (m).
        """
        if not self.active:
            return None
        self.t += dt
        self._advance(dt, v_est, steer_to_delta(steer_now, self.p))
        if obs_list:
            self._observe(obs_list, self.t)

        if self.m is None:
            self.message = "searching for the marker"
            return 0.0, self.model.pwm_for_speed(0.16)          # creep forward looking for it

        if self.last_seen is not None and self.t - self.last_seen > self.lost_timeout and self.best_size < 90:
            self.state, self.message = FAILED, "lost the marker"
            return None

        mx, my = self._to_vehicle(self.m[0], self.m[1])
        facing = wrap(self.m[2] - self.oth)                       # marker's outward direction, vehicle frame
        ax, ay = -math.cos(facing), -math.sin(facing)             # axis pointing into the bay
        psi_e = wrap(math.atan2(ay, ax))                          # bay heading relative to the car

        # Pure pursuit toward a point on the bay's centre line, a lookahead in front of the car.
        # d_R: how far out from the marker (along the axis) the rear axle's projection is.
        d_R = (0.0 - mx) * (-ax) + (0.0 - my) * (-ay)
        rx, ry = 0.0 - mx, 0.0 - my
        cross = abs(ax * ry - ay * rx)
        lookahead = min(0.9, max(0.32, 0.30 + 0.9 * cross))
        d_c = max(d_R - lookahead, self.p.front_x + self.stop_gap * 0.5)
        cx, cy = mx - ax * d_c, my - ay * d_c
        l_d = max(math.hypot(cx, cy), 0.12)
        alpha = math.atan2(cy, cx)
        delta = math.atan2(2.0 * self.p.wheelbase * math.sin(alpha), l_d)
        delta = max(steer_to_delta(-1.0, self.p), min(steer_to_delta(1.0, self.p), delta))
        steer = delta_to_steer(delta, self.p)

        along = (mx - self.p.front_x) * ax + my * ay              # distance from the bumper to the marker
        gap_now = min(front_gap, max(along, 0.0))
        if gap_now <= self.stop_gap:
            self.state, self.message = DONE, "parked"
            return 0.0, 0.0
        v = min(self.v_max, max(self.v_min, 0.8 * (gap_now - self.stop_gap) + self.v_min))
        self.message = f"approaching, {along * 100:.0f} cm to go"
        return steer, self.model.pwm_for_speed(v)

    def error(self, car_x, car_y, car_theta, marker):
        """Final pose error against the bay's true centre line (for evaluation only)."""
        ax, ay = -math.cos(marker.yaw), -math.sin(marker.yaw)
        dx, dy = car_x - marker.x, car_y - marker.y
        lateral = ax * dy - ay * dx
        yaw = wrap(car_theta - math.atan2(ay, ax))
        return lateral, yaw
