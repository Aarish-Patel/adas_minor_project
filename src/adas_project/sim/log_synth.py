"""Make a drive log from the simulator, in exactly the format the car writes (pi/drive_log.py).

    python -m sim.log_synth out.jsonl.gz [seconds]

The simulated car has KNOWN dynamics, so fitting this log (sim/log_fit.py) must give those values
back - that checks the whole log -> trajectory -> fit chain without needing the car.

Conventions reproduced from the real car:
  * scan angles are RAW sensor angles: clockwise-positive, rotated by the mount's yaw offset
  * servo: raw degrees, above the centre = turning right
  * motor: wire value, negative = physical forward (motor_reversed)
"""
import gzip
import json
import math
import sys

import numpy as np

from .lidar_sim import LidarSim
from .real_car import SERVO_TRAVEL_DEG, real_profile
from .simulator import Simulator
from .world import Box, Cone, Wall, World

SERVO_CENTRE = 87.0
YAW_OFFSET = 63.4


def arena():
    w = World()
    for a, b, c, d in ((-1.5, -2.0, 5.5, -2.0), (5.5, -2.0, 5.5, 2.0), (5.5, 2.0, -1.5, 2.0), (-1.5, 2.0, -1.5, -2.0)):
        w.add(Wall(a, b, c, d))
    # clutter so scan matching has features in every direction (like a real room)
    for x, y in ((1.0, 1.3), (2.5, -1.4), (3.8, 1.1), (0.2, -1.5), (4.6, -0.6)):
        w.add(Box(x, y, 0.3, 0.25, 0.4))
    for x, y in ((2.0, 0.9), (3.2, -0.8)):
        w.add(Cone(x, y, 0.05))
    return w


def script(t):
    """Command schedule: throttle steps at several PWMs (incl. coast-downs), and steering both ways."""
    # every forward leg is followed by the same leg in reverse, so the car stays inside the room
    seq = [  # (duration, pwm_physical, steer_stick -1..1  (+ = left, as in the simulator))
        (1.0, 0, 0.0), (2.2, 110, 0.0), (1.2, 0, 0.0), (2.2, -110, 0.0), (1.2, 0, 0.0),
        (2.0, 140, 0.0), (1.2, 0, 0.0), (2.0, -140, 0.0), (1.2, 0, 0.0),
        (2.5, 120, 0.25), (1.2, 0, 0.0), (2.5, -120, 0.25), (1.2, 0, 0.0),
        (2.5, 120, -0.25), (1.2, 0, 0.0), (2.5, -120, -0.25), (1.2, 0, 0.0),
        (1.6, 170, 0.0), (1.4, 0, 0.0), (1.6, -170, 0.0), (1.4, 0, 0.0),
        (2.5, 130, 0.12), (1.0, 0, 0.0), (2.5, -130, 0.12), (1.5, 0, 0.0),
    ]
    acc = 0.0
    for dur, pwm, st in seq:
        if t < acc + dur:
            return st, pwm
        acc += dur
    return 0.0, 0


def stick_to_servo(s):
    """Simulator stick (+ = left) -> raw servo angle (above centre = right) on the real car."""
    return SERVO_CENTRE - s * (SERVO_TRAVEL_DEG[1] if s > 0 else SERVO_TRAVEL_DEG[0])


def generate(path, seconds=None, seed=3, start=(0.0, 0.0, 0.0)):
    rp = real_profile()
    world = arena()
    sim = Simulator(world, start, adas_on="off", params=rp.params, aeb_config=rp.aeb, dynamics=rp.dynamics,
                    adas_speed_model=rp.speed_model, lidar=LidarSim(seed=seed, **rp.lidar_kw), seed=seed)
    total = seconds or 43.0
    t0 = 1.7e9
    last_scan, last_cmd = -1, None
    truth = []
    with gzip.open(path, "wt") as f:
        def rec(r):
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
        rec({"k": "meta", "t": t0, "source": "synthetic", "version": 1, "git": None,
             "tuning": {"mount": {"yaw_offset_deg": YAW_OFFSET, "min_valid_range_m": 0.2},
                        "servo": {"left_center": SERVO_CENTRE, "right_center": SERVO_CENTRE, "motor_reversed": True}},
             "truth": {"v_max": rp.dynamics.speed_model.v_max, "deadband": rp.dynamics.speed_model.deadband,
                       "tau_motor": rp.dynamics.tau_motor, "coast_decel": rp.dynamics.coast_decel,
                       "k_curv_per_deg": 0.068, "servo_centre": SERVO_CENTRE}})
        while sim.t < total:
            steer, pwm = script(sim.t)
            cmd = (round(stick_to_servo(steer)), -int(pwm))
            if cmd != last_cmd or int(sim.t / 0.05) != int((sim.t - 0.01) / 0.05):   # resend at 20 Hz like the controller
                rec({"k": "cmd", "t": t0 + sim.t, "line": f"A {cmd[0]} {cmd[0]}"})
                rec({"k": "cmd", "t": t0 + sim.t, "line": f"M {cmd[1]}"})
                last_cmd = cmd
            sim.step(steer, pwm)
            if sim.scan_id != last_scan and sim.last_scan is not None:
                last_scan = sim.scan_id
                ts, ang, rng, ok = sim.last_scan
                a_cw = (-np.degrees(ang[ok])) % 360.0              # simulator is CCW, the RPLIDAR is clockwise
                raw = (a_cw + YAW_OFFSET) % 360.0
                rec({"k": "scan", "t": t0 + ts, "a": [int(round(x * 100)) for x in raw],
                     "d": [int(round(x * 1000)) for x in rng[ok]], "q": [15] * int(ok.sum())})
            truth.append((sim.t, sim.car.x, sim.car.y, sim.car.theta, sim.car.v))
    return np.array(truth)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_log.jsonl.gz"
    generate(out, float(sys.argv[2]) if len(sys.argv) > 2 else None)
    print("wrote", out)
