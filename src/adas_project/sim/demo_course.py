"""The demo course (TODO AL): one geometry for the twin world, the printable parts (tools/demo_course_parts.py) and the guide
(docs/demo_course.md).

A 5.0 x 2.4 m floor (1 m of it behind the start, room for the return-to-start turn), ideally along a real wall (the wall is the parking kerb and the brake wall). Frame: the start mark, x ahead,
y left (the relay's world frame after ORIGIN at the start). Stations, in the order they are shown:

  S  start / return-to-start mark                       (0, 0) facing +x
  Z  speed zone 10 km/h (drawn on the dashboard)         x 0.4..1.2, y -0.5..0.5
  O  obstacle to plan round (a cardboard box)            centre (1.8, 0.45), 0.30 x 0.30
  G  click-to-go goal                                    (3.0, 0.2): one left turn on from where the car leaves the parking
     space
  T  thin post (printed, 16 mm) beside the route         (2.0, 1.05), between the box and the side wall
  P  parking lane along the wall, two parked "cars"      lane x 0.1..3.1, y -1.17..-0.77; cars at x 0.4..0.73 and 2.23..2.56,
     5 cm off the wall
     (shoe-box size cardboard boxes), the gap between them 1.5 m (4.5 car lengths: easy, repeatable)
  B  brake wall (the room's end wall)                    x = 4.0
  D  optional dark box (black felt, off the route)       centre (3.75, 1.0), 0.25 x 0.25, the far corner - only for the
     "LiDAR blind" demo (twin world demo_dark; the scripted run uses demo, without it)

The far end (x 2.6..4.0) is kept free: leaving the parking space the car turns left there, and round for the return to start
(its turning circle is ~1.3 m; with the lane 0.5 m further on, the car had only 1.1 m to the end wall and looped beside the walls).

Everything the LiDAR must see has to reach above its scan plane: LIDAR_PLANE_M (measure it on the car; default 0.20 m) + 3 cm.
"""
ROOM = (-1.0, 4.0, -1.2, 1.2)                 # x0, x1, y0, y1: the floor (walls in the twin)
START = (0.0, 0.0, 0.0)
SPEED_ZONE = {"kind": "rect", "x0": 0.4, "y0": -0.5, "x1": 1.2, "y1": 0.5, "kph": 10.0}
OBSTACLE = (1.8, 0.45, 0.30, 0.30)            # cx, cy, length (x), width (y)
GOAL = (3.0, 0.2)
THIN_POST = (2.0, 1.05, 0.008)                 # cx, cy, radius (16 mm post)
LANE = ((0.1, -0.77), (3.1, -1.17))           # road-edge corner, kerb corner: drawn with the Parking lane tool, its kerb edge 3 cm in front
                                              # of the wall (on the wall itself, a slightly rotated map put wall returns inside the lane)
PARKED = [(0.565, -1.05, 0.33, 0.20), (2.395, -1.05, 0.33, 0.20)]   # cx, cy, length, width: 5 cm off the wall; the gap is 1.50 m
PARK_CLICK = (1.48, -0.97)                     # click in the gap
DARK_BOX = (3.75, 1.0, 0.25, 0.25)
LIDAR_PLANE_M = 0.20                          # height of the car's LiDAR scan plane above the floor (measure on the car)
MIN_OBJECT_HEIGHT_M = LIDAR_PLANE_M + 0.03


def world(dark=True):
    """The twin world: the room walls, the obstacle, the parked cars, the thin post (and the dark box)."""
    from sim.hw_worlds import _finish, room_walls
    from sim.world import Box, Cone, World
    w = World()
    room_walls(w, *ROOM)
    w.add(Box(*OBSTACLE))
    for c in PARKED:
        w.add(Box(*c))
    w.add(Cone(*THIN_POST))
    if dark:
        d = Box(*DARK_BOX)
        d.dark = True                                             # black felt: no LiDAR return
        w.add(d)
    return _finish(w), START
