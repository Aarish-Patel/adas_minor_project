"""The scenarios you can pick in the viewer.

Each entry builds a world and says how the car is driven: 'manual' (keyboard or
gamepad), 'virtual' (the human-like virtual driver, with attention lapses) or a
'script' (fixed inputs).
"""

import math

import numpy as np

from .world import Box, Cone, FloorLine, Marker, MovingBox, MovingCircle, Wall, World


def _room(w, x0, x1, y0, y1, h=0.25):
    w.add(Wall(x0, y0, x1, y0, h))
    w.add(Wall(x1, y0, x1, y1, h))
    w.add(Wall(x1, y1, x0, y1, h))
    w.add(Wall(x0, y1, x0, y0, h))


def playground():
    w = World()
    _room(w, -0.8, 5.2, -1.8, 1.8)
    for x, y in ((1.3, 0.2), (2.3, -0.4), (3.3, 0.5), (4.1, -0.8)):
        w.add(Cone(x, y, 0.03))
    w.add(Box(2.9, 1.15, 0.5, 0.22, 0.15))
    w.add(Box(1.6, -1.15, 0.35, 0.2, -0.3))
    w.add(MovingCircle(2.0, 1.4, 0.0, -0.13, 0.05))
    return w, (0.0, 0.0, 0.0)


def wall_stop():
    w = World()
    _room(w, -1.0, 4.6, -1.0, 1.0)
    w.add(Box(4.2, 0.0, 0.06, 1.6, 0.0, 0.2))
    for i, y in enumerate((-0.55, 0.55)):
        w.add(FloorLine([[-0.5, y], [4.0, y]], 0.02, "#f5f5f5", dashed=True))
    return w, (0.0, 0.0, 0.0)


def pedestrian():
    w = World()
    _room(w, -1.0, 5.0, -1.5, 1.5)
    w.add(MovingCircle(2.6, 1.3, 0.0, -0.42, 0.05))
    w.add(FloorLine([[2.35, -1.4], [2.35, 1.4]], 0.05, "#f2c94c"))
    w.add(FloorLine([[2.85, -1.4], [2.85, 1.4]], 0.05, "#f2c94c"))
    return w, (0.0, 0.0, 0.0)


def emerging():
    w = World()
    _room(w, -1.0, 5.0, -1.5, 1.5)
    w.add(Box(2.5, -0.42, 0.42, 0.24, 0.0, 0.18))
    w.add(MovingCircle(2.5, -0.9, 0.0, 0.55, 0.05))
    return w, (0.0, 0.0, 0.0)


def slalom():
    w = World()
    _room(w, -1.0, 6.0, -1.4, 1.4)
    for i in range(6):
        w.add(Cone(1.0 + i * 0.75, 0.28 if i % 2 == 0 else -0.28, 0.03))
    return w, (0.0, 0.0, 0.0)


def corridor():
    w = World()
    w.add(Wall(-1.0, 0.36, 5.5, 0.36, 0.25))
    w.add(Wall(-1.0, -0.36, 5.5, -0.36, 0.25))
    w.add(Wall(5.5, -0.36, 5.5, 0.36, 0.25))
    w.add(Box(3.0, -0.15, 0.10, 0.10, 0.0, 0.12))
    return w, (0.0, 0.0, 0.0)


def virtual_arena(seed=7):
    from .intent_data import random_world
    rng = np.random.default_rng(seed)
    w = random_world(rng)
    return w, (0.0, 0.0, 0.0)


def leader():
    w = World()
    _room(w, -1.0, 9.0, -1.2, 1.2)
    w.add(MovingBox(1.4, 0.0, 0.28, 0.14, 0.0, 0.35))
    return w, (0.0, 0.0, 0.0)


def lane_road():
    w = World()
    _room(w, -1.0, 10.0, -1.6, 1.6)
    xs = np.arange(-0.5, 9.6, 0.1)
    yc = 0.30 * np.sin(0.55 * xs)
    for off in (0.225, -0.225):
        w.add(FloorLine([[float(x), float(y + off)] for x, y in zip(xs, yc)], 0.025, "#f5f5f5", lane=True))
    w.add(FloorLine([[float(x), float(y)] for x, y in zip(xs[::2], yc[::2])], 0.012, "#facc15", dashed=True))
    return w, (0.0, 0.0, 0.0)


def cutin():
    w = World()
    _room(w, -1.0, 7.0, -1.6, 1.6)
    car = MovingBox(2.45, 1.6, 0.28, 0.14, -math.pi / 2, 0.5)
    car.colour = "#a855f7"
    w.add(car)
    w.add(FloorLine([[-0.5, 0.0], [6.5, 0.0]], 0.015, "#f5f5f5", dashed=True))
    return w, (0.0, 0.0, 0.0)


def parking():
    w = World()
    _room(w, -0.8, 5.0, -2.0, 2.0)
    # bay in the far wall, two parked cars either side, ArUco on the back wall of the bay
    w.add(Box(3.85, 0.30, 0.34, 0.16, 0.0, 0.1, style="car", colour="#dc2626"))
    w.add(Box(3.85, -0.30, 0.34, 0.16, 0.0, 0.1, style="car", colour="#f59e0b"))
    w.add(Box(4.45, 0.0, 0.06, 0.60, 0.0, 0.16))
    w.add(Marker(7, 4.415, 0.0, math.pi, z=0.09, size=0.11))
    w.add(FloorLine([[3.6, 0.22], [4.4, 0.22]], 0.02, "#f5f5f5"))
    w.add(FloorLine([[3.6, -0.22], [4.4, -0.22]], 0.02, "#f5f5f5"))
    return w, (0.0, -1.0, math.radians(20))


def signs():
    w = World()
    _room(w, -0.8, 8.0, -1.0, 1.0)
    w.add(Marker(21, 1.6, 0.5, -math.pi / 2, z=0.14, size=0.10))    # speed limit
    w.add(Marker(22, 4.4, 0.5, -math.pi / 2, z=0.14, size=0.10))    # stop
    w.add(Marker(23, 6.4, 0.5, -math.pi / 2, z=0.14, size=0.10))    # school zone
    return w, (0.0, 0.0, 0.0)


SCENARIOS = {
    "playground": {"name": "Playground", "build": playground, "driver": "manual",
                   "text": "Free driving among cones, boxes and a wandering pedestrian."},
    "wall": {"name": "Emergency stop", "build": wall_stop, "driver": "manual",
             "text": "Floor the throttle at the wall. ADAS OFF crashes; ADAS ON stops short."},
    "pedestrian": {"name": "Pedestrian crossing", "build": pedestrian, "driver": "manual",
                   "text": "A pedestrian walks across the road. The tracker predicts where they will be."},
    "emerging": {"name": "Emerging pedestrian", "build": emerging, "driver": "manual",
                 "text": "A pedestrian steps out from behind a box the LiDAR cannot see through."},
    "slalom": {"name": "Slalom", "build": slalom, "driver": "manual",
               "text": "Weave through the cones. Predicted-path warnings and speed scaling."},
    "corridor": {"name": "Narrow corridor", "build": corridor, "driver": "manual",
                 "text": "Tight walls. The car slows down as obstacles approach its sides."},
    "virtual": {"name": "Virtual driver (lapses)", "build": virtual_arena, "driver": "virtual",
                "text": "A human-like virtual driver who sometimes looks away. Watch the ADAS step in."},
    "leader": {"name": "Follow the leader", "build": leader, "driver": "manual",
               "text": "A slower car ahead. Collision avoidance keeps a safe gap."},
    "parking": {"name": "Parking bay (ArUco)", "build": parking, "driver": "manual",
                "text": "Reverse-free bay with an ArUco marker on the back wall."},
    "lane": {"name": "Lane keeping", "build": lane_road, "driver": "manual",
             "text": "Curving road. Drift off the line and the car warns you; in Assist mode it steers itself back."},
    "cutin": {"name": "Car cuts in", "build": cutin, "driver": "manual",
              "text": "Another car crosses your path. The tracker predicts where it will be."},
    "signs": {"name": "Traffic signs (ISA)", "build": signs, "driver": "manual",
              "text": "ArUco markers act as speed-limit, stop and school-zone signs."},
}
