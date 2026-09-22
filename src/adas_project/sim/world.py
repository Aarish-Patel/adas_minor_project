"""The arena: static and moving obstacles, and collision clearance for the car."""

import math

import numpy as np


class Wall:
    def __init__(self, x1, y1, x2, y2, height=0.25):
        self.seg = np.array([[x1, y1, x2, y2]], dtype=float)
        self.height = height

    def describe(self):
        x1, y1, x2, y2 = self.seg[0]
        return {"k": "wall", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "h": self.height}

    def update(self, dt):
        pass

    def segments(self):
        return self.seg

    def circles(self):
        return np.empty((0, 3))


def _box_segments(cx, cy, length, width, heading):
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    corners = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]
    pts = [(cx + c * x - s * y, cy + s * x + c * y) for x, y in corners]
    return np.array([[*pts[i], *pts[(i + 1) % 4]] for i in range(4)], dtype=float)


class Box:
    """Static rectangle (foam block, parked car, bay wall...)."""

    def __init__(self, cx, cy, length, width, heading=0.0, height=0.14, style="box", colour="#2563eb"):
        self.seg = _box_segments(cx, cy, length, width, heading)
        self.geo = (cx, cy, length, width, heading, height)
        self.style, self.colour = style, colour

    def describe(self):
        cx, cy, l, w, hd, h = self.geo
        return {"k": "box", "cx": cx, "cy": cy, "l": l, "w": w, "hd": hd, "h": h, "style": self.style, "c": self.colour}

    def update(self, dt):
        pass

    def segments(self):
        return self.seg

    def circles(self):
        return np.empty((0, 3))


class Cone:
    """Static round obstacle (traffic cone)."""

    def __init__(self, x, y, radius=0.03):
        self.c = np.array([[x, y, radius]], dtype=float)

    def describe(self):
        x, y, r = self.c[0]
        return {"k": "cone", "x": x, "y": y, "r": r}

    def update(self, dt):
        pass

    def segments(self):
        return np.empty((0, 4))

    def circles(self):
        return self.c


class MovingCircle:
    """Pedestrian-like object walking at constant velocity."""
    moving = True

    def __init__(self, x, y, vx, vy, radius=0.05):
        self.x, self.y, self.vx, self.vy, self.r = x, y, vx, vy, radius

    def describe(self):
        return {"k": "ped", "r": self.r}

    def dynamic(self):
        return {"x": self.x, "y": self.y, "hd": math.atan2(self.vy, self.vx) if (self.vx or self.vy) else 0.0}

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def segments(self):
        return np.empty((0, 4))

    def circles(self):
        return np.array([[self.x, self.y, self.r]])


class MovingBox:
    """Leader car driving in a straight line."""
    moving = True

    def __init__(self, cx, cy, length, width, heading, speed):
        self.cx, self.cy, self.length, self.width = cx, cy, length, width
        self.heading, self.speed = heading, speed

    def describe(self):
        return {"k": "leader", "l": self.length, "w": self.width, "c": getattr(self, "colour", "#2563eb")}

    def dynamic(self):
        return {"x": self.cx, "y": self.cy, "hd": self.heading}

    def update(self, dt):
        self.cx += self.speed * math.cos(self.heading) * dt
        self.cy += self.speed * math.sin(self.heading) * dt

    def segments(self):
        return _box_segments(self.cx, self.cy, self.length, self.width, self.heading)

    def circles(self):
        return np.empty((0, 3))


class Marker:
    """Printed ArUco marker (flat: invisible to the LiDAR, visible to the camera)."""

    def __init__(self, marker_id, x, y, yaw, z=0.09, size=0.10):
        self.marker_id, self.x, self.y, self.yaw, self.z, self.size = marker_id, x, y, yaw, z, size

    def update(self, dt):
        pass

    def segments(self):
        return np.empty((0, 4))

    def circles(self):
        return np.empty((0, 3))

    def describe(self):
        return {"k": "marker", "id": self.marker_id, "x": self.x, "y": self.y, "yaw": self.yaw,
                "z": self.z, "size": self.size}


class FloorLine:
    """Tape or paint on the floor (camera only)."""

    def __init__(self, pts, width=0.02, colour="#f5f5f5", dashed=False, lane=False):
        self.pts, self.width, self.colour, self.dashed = pts, width, colour, dashed
        self.lane = lane

    def update(self, dt):
        pass

    def segments(self):
        return np.empty((0, 4))

    def circles(self):
        return np.empty((0, 3))

    def describe(self):
        return {"k": "line", "pts": self.pts, "w": self.width, "c": self.colour, "dashed": self.dashed}


class World:
    def __init__(self):
        self.objects = []
        self._cache = None

    def add(self, obj):
        obj.id = len(self.objects)
        self.objects.append(obj)
        self._cache = None
        return obj

    def describe_static(self):
        return [dict(o.describe(), id=o.id) for o in self.objects]

    def describe_dynamic(self):
        return [dict(o.dynamic(), id=o.id) for o in self.objects if getattr(o, "moving", False)]

    def add_room(self, xmin, xmax, ymin, ymax):
        self.add(Wall(xmin, ymin, xmax, ymin))
        self.add(Wall(xmax, ymin, xmax, ymax))
        self.add(Wall(xmax, ymax, xmin, ymax))
        self.add(Wall(xmin, ymax, xmin, ymin))

    def update(self, dt):
        moved = False
        for o in self.objects:
            if getattr(o, "moving", False):
                o.update(dt)
                moved = True
        if moved:
            self._cache = None

    def _build(self):
        if self._cache is None:
            seg = [o.segments() for o in self.objects]
            circ = [o.circles() for o in self.objects]
            seg = [a for a in seg if len(a)]
            circ = [a for a in circ if len(a)]
            self._cache = (np.vstack(seg) if seg else np.empty((0, 4)),
                           np.vstack(circ) if circ else np.empty((0, 3)))
        return self._cache

    def segments(self):
        return self._build()[0]

    def circles(self):
        return self._build()[1]

    def clearance(self, x, y, theta, p):
        """Smallest gap (m) between the car's body and any obstacle. <= 0 means contact."""
        c, s = math.cos(theta), math.sin(theta)
        xmin, xmax = p.rear_x, p.front_x
        ymin, ymax = -p.width / 2.0, p.width / 2.0

        best = math.inf

        seg = self.segments()
        if len(seg):
            def to_local(px, py):
                dx, dy = px - x, py - y
                return c * dx + s * dy, -s * dx + c * dy

            x1, y1 = to_local(seg[:, 0], seg[:, 1])
            x2, y2 = to_local(seg[:, 2], seg[:, 3])

            d = np.minimum(_point_rect_dist(x1, y1, xmin, xmax, ymin, ymax),
                           _point_rect_dist(x2, y2, xmin, xmax, ymin, ymax))
            for cx, cy in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)):
                d = np.minimum(d, _point_seg_dist(cx, cy, x1, y1, x2, y2))
            d = np.where(_seg_hits_rect(x1, y1, x2, y2, xmin, xmax, ymin, ymax), 0.0, d)
            best = min(best, float(d.min()))

        circ = self.circles()
        if len(circ):
            dx, dy = circ[:, 0] - x, circ[:, 1] - y
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            d = _point_rect_dist(lx, ly, xmin, xmax, ymin, ymax) - circ[:, 2]
            best = min(best, float(d.min()))

        return best


def _point_rect_dist(px, py, xmin, xmax, ymin, ymax):
    dx = np.maximum(np.maximum(xmin - px, 0.0), px - xmax)
    dy = np.maximum(np.maximum(ymin - py, 0.0), py - ymax)
    return np.hypot(dx, dy)


def _point_seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 0, l2, 1.0)
    t = np.clip(np.where(l2 > 0, ((px - x1) * dx + (py - y1) * dy) / safe, 0.0), 0.0, 1.0)
    return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _seg_hits_rect(x1, y1, x2, y2, xmin, xmax, ymin, ymax):
    """Liang-Barsky segment vs axis-aligned rectangle test, vectorised."""
    dx, dy = x2 - x1, y2 - y1
    t0 = np.zeros_like(x1)
    t1 = np.ones_like(x1)
    ok = np.ones(x1.shape, dtype=bool)
    for p, q in ((-dx, x1 - xmin), (dx, xmax - x1), (-dy, y1 - ymin), (dy, ymax - y1)):
        parallel = p == 0
        ok &= ~(parallel & (q < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = q / p
        t0 = np.where((p < 0) & ~parallel, np.maximum(t0, r), t0)
        t1 = np.where((p > 0) & ~parallel, np.minimum(t1, r), t1)
    return ok & (t0 <= t1)
