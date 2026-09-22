"""Run every scenario with the ADAS off and on, and with a mismatched real car.

    python -m sim.run_scenarios          (from the RC_Car folder)

'Mismatch' makes the simulated car faster / slower than the speed model the
ADAS believes in (+/-25 %), to show the braking still works when the
calibration is imperfect.
"""

import sys
from dataclasses import replace

from adas.aeb import SpeedModel

from .car_sim import Dynamics
from .scenarios import default_suite
from .simulator import Simulator


def run_one(scn, adas_on, v_scale=1.0):
    world, start = scn.build()
    dyn = Dynamics(speed_model=SpeedModel(v_max=1.0 * v_scale))
    sim = Simulator(world, start, adas_on=adas_on, dynamics=dyn)
    sim.run(scn.driver, scn.duration, stop_when_stopped=not scn.run_full)
    return sim


def real_crash(sim):
    return sim.car.collided and sim.car.impact_speed >= 0.05


def verdict(scn, sim):
    if scn.hazard:
        ok = not real_crash(sim)
    else:
        ok = (not sim.car.collided) and sim.brake_time == 0.0
    return ok


def fmt(sim):
    status = "CRASH" if sim.car.collided else "ok"
    impact = f" impact {sim.car.impact_speed:.2f} m/s" if sim.car.collided else ""
    return f"{status:5s} gap {max(sim.min_clearance, 0.0) * 100:5.1f} cm  brake {sim.brake_time:4.2f} s{impact}"


def main():
    failures = 0
    print(f"{'scenario':58s} | {'ADAS OFF':38s} | {'ADAS ON':38s} | ADAS ON, car 25% faster | ADAS ON, car 25% slower")
    for scn in default_suite():
        off = run_one(scn, False)
        on = run_one(scn, True)
        fast = run_one(scn, True, 1.25)
        slow = run_one(scn, True, 0.75)

        results = [verdict(scn, s) for s in (on, fast, slow)]
        failures += results.count(False)
        marks = ["PASS" if r else "FAIL" for r in results]
        print(f"{scn.name:58s} | {fmt(off):38s} | {fmt(on):38s} | {marks[1]} ({fmt(fast)}) | {marks[2]} ({fmt(slow)})"
              f"   -> {marks[0]}")

    print()
    print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
