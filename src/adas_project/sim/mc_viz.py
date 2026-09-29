"""Batch of random drives, each run twice (ADAS off / on) with identical driver input,
recorded as trajectories for the viewer's Monte Carlo lab to play back sped up.
"""

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from adas.aeb import SpeedModel

from .car_sim import Dynamics
from .monte_carlo import random_case
from .simulator import Simulator

SAMPLE = 0.1
DURATION = 8.0


def _run(seed, adas_on):
    rng = np.random.default_rng(seed)
    world, driver, pwm, scale, kind = random_case(rng)
    dyn = Dynamics(speed_model=SpeedModel(v_max=1.0 * scale))
    sim = Simulator(world, (0.0, 0.0, 0.0), adas_on=adas_on, dynamics=dyn, seed=seed)

    traj, dynpos = [], []
    next_sample = 0.0
    while sim.t < DURATION and not sim.car.collided:
        steer, p = driver(sim.t, sim)
        sim.step(steer, p)
        if sim.t >= next_sample:
            next_sample += SAMPLE
            c = sim.car
            traj.append([round(sim.t, 2), round(c.x, 3), round(c.y, 3), round(c.theta, 3), sim.level])
            if adas_on:
                dynpos.append([[round(d["x"], 3), round(d["y"], 3)] for d in world.describe_dynamic()])
    crashed = bool(sim.car.collided and sim.car.impact_speed >= 0.05)
    c = sim.car
    traj.append([round(sim.t, 2), round(c.x, 3), round(c.y, 3), round(c.theta, 3), sim.level])
    return traj, dynpos, crashed, float(c.impact_speed), world, pwm, scale, kind


def run_case(seed):
    off, _, off_crash, off_v, w_off, pwm, scale, kind = _run(seed, False)
    on, dynpos, on_crash, on_v, world, _, _, _ = _run(seed, True)
    # static objects as they were at the start, moving ones described separately
    static = [o for o in world.describe_static() if o["k"] not in ("ped", "leader")]
    movers = [o["k"] for o in world.describe_static() if o["k"] in ("ped", "leader")]
    return {"seed": seed, "pwm": pwm, "scale": round(scale, 2), "kind": str(kind),
            "static": static, "movers": movers, "off": off, "on": on, "dyn": dynpos,
            "off_crash": off_crash, "on_crash": on_crash, "off_impact": round(off_v, 2), "on_impact": round(on_v, 2)}


def run_batch(n=16, seed0=None):
    import time
    seed0 = seed0 if seed0 is not None else int(time.time()) % 100000
    with ProcessPoolExecutor() as pool:
        return list(pool.map(run_case, range(seed0, seed0 + n)))


if __name__ == "__main__":
    import json
    import time
    t0 = time.time()
    cases = run_batch(8, 1000)
    print(f"8 cases in {time.time() - t0:.1f} s, payload {len(json.dumps(cases)) // 1024} KB")
    print([(c["off_crash"], c["on_crash"]) for c in cases])
