"""Quick regression check of the important behaviours (about a minute).

    python -m sim.selftest

Run it after changing anything in adas/ or the tuning. Every line should say ok.
"""

import math
import sys
import time

from adas.aeb import SpeedModel
from adas.servo import ServoCalibration, command_text, steering_to_servo_angles
from adas.vehicle_params import VehicleParams, delta_to_steer, steer_to_delta

from .library import cutin, lane_road, leader, parking, signs, wall_stop
from .scenarios import head_on_wall, pedestrian_crossing
from .simulator import Simulator
from .world import Marker

checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))


def wall(adas, pwm=255):
    scn = head_on_wall(pwm)
    world, start = scn.build()
    sim = Simulator(world, start, adas_on=adas)
    sim.run(scn.driver, 6.0)
    return sim


def main():
    t0 = time.time()
    p = VehicleParams()

    # geometry and conventions
    check("steering maps to servos symmetrically at centre",
          steering_to_servo_angles(0.0, ServoCalibration()) == (90.0, 90.0))
    inner_left = steering_to_servo_angles(0.6, ServoCalibration())
    check("Ackermann: inner wheel turns more than outer",
          abs(inner_left[0] - 90) > abs(inner_left[1] - 90), f"left {inner_left[0]:.1f}, right {inner_left[1]:.1f}")
    check("steer <-> bicycle angle round trip", abs(delta_to_steer(steer_to_delta(0.37, p), p) - 0.37) < 1e-3)
    check("command text has servo and motor lines", command_text(0.0, 100, ServoCalibration()).count("\n") == 2)

    # collision avoidance
    off, on = wall(False), wall(True)
    check("without ADAS the car hits the wall at full throttle", off.car.collided and off.car.impact_speed > 0.5)
    check("with ADAS it stops short", not on.car.collided and on.min_clearance > 0.05,
          f"{on.min_clearance * 100:.0f} cm gap")

    scn = pedestrian_crossing(180, 0.4)
    world, start = scn.build()
    sim = Simulator(world, start, adas_on=True)
    sim.run(scn.driver, 6.0)
    check("crossing pedestrian is avoided", not (sim.car.collided and sim.car.impact_speed > 0.05))

    world, start = cutin()
    sim = Simulator(world, start, adas_on=True)
    while sim.t < 8 and not sim.car.collided:
        sim.step(0.0, 200 if sim.t > 0.3 else 0)
    check("car cutting in is avoided", not sim.car.collided)

    # fails safe
    world, start = wall_stop()
    sim = Simulator(world, start, adas_on="active")
    while sim.t < 8 and not sim.car.collided:
        sim.lidar_enabled = sim.t < 1.0
        sim.step(0.0, 220 if sim.t > 0.3 else 0)
    check("LiDAR unplugged at speed: watchdog stops the car", not sim.car.collided and abs(sim.car.v) < 0.05)

    # assists
    world, start = parking()
    sim = Simulator(world, (0.4, -0.6, math.radians(10)), adas_on="active")
    sim.adas.parking.start()
    marker = next(o for o in world.objects if isinstance(o, Marker))
    while sim.t < 40 and sim.adas.parking.state == "approach" and not sim.car.collided:
        sim.step(0.0, 0.0)
    for _ in range(80):
        sim.step(0.0, 0.0)
    lat, yaw = sim.adas.parking.error(sim.car.x, sim.car.y, sim.car.theta, marker)
    check("auto-park ends in the bay, centred and straight",
          sim.adas.parking.state == "done" and abs(lat) < 0.03 and abs(math.degrees(yaw)) < 6 and not sim.car.collided,
          f"{lat * 100:.1f} cm, {math.degrees(yaw):.1f} deg")

    world, start = signs()
    sim = Simulator(world, start, adas_on="active")
    stopped = False
    while sim.t < 20 and not sim.car.collided:
        sim.step(0.0, 200)
        stopped = stopped or (sim.adas.isa.active or "").startswith("STOP") and abs(sim.car.v) < 0.03
    check("traffic signs: stops at the STOP sign", stopped)

    world, start = leader()
    sim = Simulator(world, start, adas_on="active")
    sim.adas.acc.enabled = True
    while sim.t < 14 and not sim.car.collided:
        sim.step(0.0, 220)
    check("follow-the-leader keeps a gap without touching", not sim.car.collided and sim.min_clearance > 0.1,
          f"min gap {sim.min_clearance * 100:.0f} cm")

    world, start = lane_road()
    sim = Simulator(world, start, adas_on="active")
    sim.adas.lane.mode = "assist"
    worst = 0.0
    while sim.t < 25 and sim.car.x < 9.0 and not sim.car.collided:
        sim.step(0.0, 125 if sim.t > 0.3 else 0)
        worst = max(worst, abs(sim.car.y - 0.30 * math.sin(0.55 * sim.car.x)))
    check("lane-keeping assist stays inside the lane", worst < 0.10, f"worst drift {worst * 100:.0f} cm")

    # pi runtime against the simulator
    from pi.hil import run as hil_run
    r = hil_run("wall", verbose=False)
    check("Pi runtime + emulated ESP32 stops at the wall", not r["crashed"] and r["servo_protocol_max_error_deg"] < 0.5)

    print(f"\n{sum(checks)}/{len(checks)} ok in {time.time() - t0:.0f} s")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
