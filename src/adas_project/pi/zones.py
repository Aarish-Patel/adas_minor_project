"""Speed-limit zones (user, 28 Sep): shapes drawn on the map in the world frame, each with a limit in km/h as on a
full-size road; the car is ~1:CAR_SCALE, so the limit is scaled down the same way (50 km/h -> ~1.0 m/s, full speed
on this car; 20 km/h -> 0.4 m/s; 10 km/h -> 0.2 m/s). Outside every zone there is no limit.

The relay caps the throttle to the zone the car is in - or will be in LOOKAHEAD_S from now, so it slows before
entering, as intelligent speed assistance does.

Shapes (world frame, metres, the relay's odometry origin):
  {"kind": "rect", "x0": .., "y0": .., "x1": .., "y1": .., "kph": 20}
  {"kind": "circle", "x": .., "y": .., "r": .., "kph": 10}
  {"kind": "poly", "pts": [[x, y], ...], "kph": 15}
"""
import json
import math

CAR_SCALE = 14.0
LOOKAHEAD_S = 0.5


def kph_to_car(kph):
    """A full-size speed limit (km/h) as the scaled-down speed for this car (m/s)."""
    return float(kph) / 3.6 / CAR_SCALE


def car_to_kph(v):
    return float(v) * CAR_SCALE * 3.6


def _in_poly(x, y, pts):
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def contains(z, x, y):
    k = z.get("kind")
    if k == "rect":
        return min(z["x0"], z["x1"]) <= x <= max(z["x0"], z["x1"]) and min(z["y0"], z["y1"]) <= y <= max(z["y0"], z["y1"])
    if k == "circle":
        return math.hypot(x - z["x"], y - z["y"]) <= z["r"]
    if k == "poly":
        return len(z.get("pts", [])) >= 3 and _in_poly(x, y, z["pts"])
    return False


class SpeedZones:
    def __init__(self):
        self.zones = []

    def set(self, zones):
        """Replace all zones (list of dicts, or a JSON string); malformed entries are dropped."""
        if isinstance(zones, str):
            zones = json.loads(zones)
        ok = []
        for z in zones or []:
            try:
                z = dict(z)
                z["kph"] = float(z["kph"])
                if z["kph"] > 0 and z.get("kind") in ("rect", "circle", "poly"):
                    contains(z, 0.0, 0.0)          # validates the fields
                    ok.append(z)
            except (KeyError, TypeError, ValueError):
                continue
        self.zones = ok
        return len(ok)

    def limit_kph(self, x, y):
        """The lowest limit of the zones containing (x, y), or None."""
        lims = [z["kph"] for z in self.zones if contains(z, x, y)]
        return min(lims) if lims else None

    def limit_ahead(self, pose, v):
        """Limit (km/h, or None) at the car's position and where it will be LOOKAHEAD_S from now."""
        x, y, th = pose
        ahead = v * LOOKAHEAD_S
        lims = [l for l in (self.limit_kph(x, y), self.limit_kph(x + ahead * math.cos(th), y + ahead * math.sin(th)))
                if l is not None]
        return min(lims) if lims else None
