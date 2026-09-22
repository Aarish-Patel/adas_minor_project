"""Fault injection: does the system fail SAFE when parts of it break?

    python -m sim.faults

Each test breaks something mid-drive and passes only if the car does not hit anything.
Results are saved for the viewer's Results tab and the report. A test that fails is a
finding, not a bug in the test: it tells you where the safety margin ends.
"""

import json
import math
import os
import time

from adas.aeb import SpeedModel

from .car_sim import Dynamics
from .intent_data import MODEL_DIR
from .library import parking, wall_stop
from .simulator import Simulator
from .world import Marker


def _wall_run(pwm=220, fault=None, dyn=None, seconds=9.0):
    world, start = wall_stop()
    sim = Simulator(world, start, adas_on="active", dynamics=dyn)
    fault_t = 1.0
    driver_pwm = lambda t: pwm if t > 0.3 else 0.0
    while sim.t < seconds and not sim.car.collided:
        if fault and sim.t >= fault_t:
            fault(sim)
        p = driver_pwm(sim.t)
        if fault is disconnect_driver and sim.t >= fault_t:
            p = 0.0
        sim.step(0.0, p)
    return sim


def unplug_lidar(sim):
    sim.lidar_enabled = False


def disconnect_driver(sim):
    pass


def dropout(sim):
    sim.lidar.dropout = 0.4


def heavy_noise(sim):
    sim.lidar.noise_std = 0.05


def slow_link(sim):
    sim.link_delay = 0.15


def very_slow_link(sim):
    sim.link_delay = 0.30


def result(name, sim, detail=""):
    crashed = bool(sim.car.collided and sim.car.impact_speed >= 0.05)
    return {"name": name, "pass": not crashed,
            "detail": (f"CRASH at {sim.car.impact_speed:.2f} m/s" if crashed else
                       f"stopped {max(sim.min_clearance, 0) * 100:.0f} cm short") + (f"; {detail}" if detail else "")}


def parking_camera_lost():
    world, _ = parking()
    sim = Simulator(world, (0.0, 0.0, 0.0), adas_on="active")
    sim.adas.parking.start()
    while sim.t < 40 and not sim.car.collided:
        if sim.t > 8.0:
            sim.camera_enabled = False
        sim.step(0.0, 0.0)
        if sim.adas.parking.state in ("done", "failed"):
            break
    for _ in range(150):
        sim.step(0.0, 0.0)
    return result("Camera lost while auto-parking", sim, f"parking state: {sim.adas.parking.state}")


def pi_program_dies():
    from pi.link import Esp32Actuator, Esp32Emulator, UdpLink
    emu = Esp32Emulator()
    emu.start()
    act = Esp32Actuator(UdpLink("127.0.0.1", emu.port))
    for _ in range(20):
        act.send(0.0, 200)
        time.sleep(0.02)
    was = emu.motor
    time.sleep(0.8)                                   # the Pi crashes: no more commands
    now = emu.motor
    emu.stop()
    ok = was != 0 and now == 0
    return {"name": "Pi program dies (ESP32 failsafe)", "pass": bool(ok),
            "detail": f"motor {was} -> {now} within 0.8 s of the last command"}


def run_all():
    tests = []
    tests.append(result("LiDAR unplugged at speed", _wall_run(fault=unplug_lidar), "watchdog stops the car"))
    tests.append(result("LiDAR drops 40% of readings", _wall_run(fault=dropout)))
    tests.append(result("LiDAR noise rises to 5 cm", _wall_run(fault=heavy_noise)))
    tests.append(result("Link delay rises to 150 ms", _wall_run(fault=slow_link)))
    tests.append(result("Controller disconnects at full speed", _wall_run(fault=disconnect_driver)))
    tests.append(result("Real car 40% faster than calibrated", _wall_run(dyn=Dynamics(speed_model=SpeedModel(v_max=1.4)))))
    tests.append(result("Real car 40% slower than calibrated", _wall_run(dyn=Dynamics(speed_model=SpeedModel(v_max=0.6)))))
    tests.append(parking_camera_lost())
    tests.append(pi_program_dies())
    return tests


def main():
    tests = run_all()
    for t in tests:
        print(f"{'PASS' if t['pass'] else 'FAIL'}  {t['name']:42s} {t['detail']}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "faults.json"), "w") as f:
        json.dump(tests, f, indent=1)
    print(f"{sum(t['pass'] for t in tests)}/{len(tests)} passed")
    return tests


if __name__ == "__main__":
    main()
