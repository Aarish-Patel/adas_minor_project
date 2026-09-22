"""Intelligent Speed Adaptation: ArUco markers acting as road signs.

    id 21  speed limit    (cap while passing)
    id 22  stop           (stop just before the sign, wait, then carry on)
    id 23  school zone    (lower cap for a stretch)

Sign positions are remembered in a dead-reckoned frame, so they keep working after
they have left the camera's view.
"""

import math

from .vehicle_params import steer_to_delta

SPEED_LIMIT, STOP, SCHOOL = 21, 22, 23


class SpeedAdaptation:
    def __init__(self, params, limit_speed=0.45, school_speed=0.30, stop_wait=3.0, stop_decel=1.0,
                 stop_offset=0.28):
        self.p = params
        self.limit_speed = limit_speed
        self.school_speed = school_speed
        self.stop_wait = stop_wait
        self.stop_decel = stop_decel
        self.stop_offset = stop_offset
        self.enabled = True
        self.reset()

    def reset(self):
        self.ox = self.oy = self.oth = 0.0
        self.signs = {}              # id -> {"pos": (X, Y), "timer": float, "done": bool}
        self.active = None           # text shown in the UI
        self.cap = math.inf

    def _to_odom(self, x, y):
        c, s = math.cos(self.oth), math.sin(self.oth)
        return self.ox + c * x - s * y, self.oy + s * x + c * y

    def update(self, dt, obs_list, v_est, steer):
        omega = v_est * math.tan(steer_to_delta(steer, self.p)) / self.p.wheelbase
        mid = self.oth + 0.5 * omega * dt
        self.ox += v_est * math.cos(mid) * dt
        self.oy += v_est * math.sin(mid) * dt
        self.oth += omega * dt

        for o in obs_list:
            if o.id in (SPEED_LIMIT, STOP, SCHOOL):
                pos = self._to_odom(o.x, o.y)
                st = self.signs.setdefault(o.id, {"pos": pos, "timer": 0.0, "done": False})
                st["pos"] = pos if "seen" not in st else (
                    0.8 * st["pos"][0] + 0.2 * pos[0], 0.8 * st["pos"][1] + 0.2 * pos[1])
                st["seen"] = True

        self.cap = math.inf
        self.active = None
        if not self.enabled:
            return self.cap

        c, s = math.cos(self.oth), math.sin(self.oth)
        for sid, st in self.signs.items():
            X, Y = st["pos"]
            along = (X - self.ox) * c + (Y - self.oy) * s          # how far ahead the sign is (m)
            if sid == SPEED_LIMIT and -1.4 < along < 1.0:
                self.cap = min(self.cap, self.limit_speed)
                self.active = "Speed limit"
            elif sid == SCHOOL and -2.0 < along < 0.8:
                self.cap = min(self.cap, self.school_speed)
                self.active = "School zone"
            elif sid == STOP and not st["done"] and along > -0.3:
                target = along - self.stop_offset
                if target > 0.02:
                    self.cap = min(self.cap, math.sqrt(2.0 * self.stop_decel * target) + 0.02)
                    self.active = "Stop ahead"
                else:
                    self.cap = 0.0
                    self.active = f"STOP  {max(0.0, self.stop_wait - st['timer']):.0f}s"
                    if abs(v_est) < 0.04:
                        st["timer"] += dt
                        if st["timer"] >= self.stop_wait:
                            st["done"] = True
        return self.cap
