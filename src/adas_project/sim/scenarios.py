"""Test scenarios. Each builds a world and a scripted 'driver'."""

import math
from dataclasses import dataclass
from typing import Callable

from .world import Box, Cone, MovingBox, MovingCircle, Wall, World


def constant_driver(steer, pwm, start_delay=0.3):
    return lambda t, sim: (steer, pwm if t >= start_delay else 0.0)


@dataclass
class Scenario:
    name: str
    build: Callable            # () -> (world, start_pose)
    driver: Callable           # (t, sim) -> (steer, pwm)
    duration: float
    hazard: bool               # True: the car must avoid something. False: it must NOT brake.
    run_full: bool = False     # keep running after the car stops (driver holds the throttle)


def head_on_wall(pwm):
    def build():
        w = World()
        w.add(Wall(3.0, -1.0, 3.0, 1.0))
        return w, (0.0, 0.0, 0.0)
    return Scenario(f"wall ahead, PWM {pwm}", build, constant_driver(0.0, pwm), 6.0, True)


def throttle_held_at_wall(pwm):
    """Driver keeps flooring it after the car has stopped: it must not creep into the wall."""
    def build():
        w = World()
        w.add(Wall(2.0, -1.0, 2.0, 1.0))
        return w, (0.0, 0.0, 0.0)
    return Scenario(f"throttle held against wall for 10 s, PWM {pwm}", build,
                    constant_driver(0.0, pwm), 10.0, True, run_full=True)


def reverse_into_wall(pwm):
    def build():
        w = World()
        w.add(Wall(-3.0, -1.0, -3.0, 1.0))
        return w, (0.0, 0.0, 0.0)
    return Scenario(f"reverse into wall, PWM -{pwm}", build, constant_driver(0.0, -pwm), 6.0, True)


def curve_into_wall(pwm, steer):
    def build():
        w = World()
        w.add(Wall(1.0, 0.5, 2.0, -0.5))
        w.add(Wall(0.0, 1.2, 2.0, 1.2))
        return w, (0.0, 0.0, 0.0)
    return Scenario(f"curving left into wall, steer {steer}, PWM {pwm}", build,
                    constant_driver(steer, pwm), 6.0, True)


def cone_beside_path(pwm, offset):
    def build():
        w = World()
        w.add(Cone(1.5, offset, 0.03))
        w.add(Wall(3.5, -1.0, 3.5, 1.0))
        return w, (0.0, 0.0, 0.0)
    return Scenario(f"cone {offset:.2f} m off the path (must not brake), PWM {pwm}", build,
                    lambda t, sim: (0.0, pwm if 0.3 <= t < 2.0 else 0.0), 3.0, False)


def pedestrian_crossing(pwm, walk_speed):
    def build():
        w = World()
        w.add(MovingCircle(2.0, 1.2, 0.0, -walk_speed, 0.05))
        return w, (0.0, 0.0, 0.0)
    return Scenario(f"pedestrian crossing at {walk_speed} m/s, PWM {pwm}", build,
                    constant_driver(0.0, pwm), 6.0, True)


def car_cuts_in(pwm, cross_speed):
    def build():
        w = World()
        w.add(MovingBox(2.45, 1.6, 0.28, 0.14, -math.pi / 2, cross_speed))
        return w, (0.0, 0.0, 0.0)
    return Scenario(f"car cuts across, {cross_speed} m/s, PWM {pwm}", build, constant_driver(0.0, pwm), 7.0, True)


def default_suite():
    suite = []
    for pwm in (100, 150, 200, 255):
        suite.append(head_on_wall(pwm))
    for pwm in (150, 255):
        suite.append(throttle_held_at_wall(pwm))
    for pwm in (120, 200):
        suite.append(reverse_into_wall(pwm))
    for pwm in (120, 180):
        suite.append(curve_into_wall(pwm, 0.5))
    for offset in (0.20, 0.30):
        suite.append(cone_beside_path(180, offset))
    for pwm in (120, 180):
        suite.append(pedestrian_crossing(pwm, 0.4))
    for pwm in (150, 200, 255):
        suite.append(car_cuts_in(pwm, 0.5))
    return suite
