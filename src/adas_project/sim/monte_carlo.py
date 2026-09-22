"""Randomised stress test: many random arenas, pedestrians, steering and speeds.

    python -m sim.monte_carlo [runs] [seed]

Reports how often the car crashes with the ADAS off and on, split by throttle
level, so weak spots (for example very high PWM) show up as numbers.
"""

import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from adas.aeb import SpeedModel

from .car_sim import Dynamics
from .simulator import Simulator
from .world import Box, Cone, MovingCircle, Wall, World


def random_case(rng):
    w = World()
    w.add_room(-1.0, 6.0, -2.0, 2.0)

    for _ in range(rng.integers(2, 6)):
        w.add(Cone(rng.uniform(1.0, 5.0), rng.uniform(-1.3, 1.3), 0.03))
    for _ in range(rng.integers(0, 3)):
        w.add(Box(rng.uniform(1.5, 5.0), rng.uniform(-1.3, 1.3), rng.uniform(0.15, 0.4),
                  rng.uniform(0.1, 0.25), rng.uniform(0, math.pi)))

    kind = rng.choice(["none", "crossing", "emerging"])
    if kind == "crossing":
        x = rng.uniform(1.5, 4.0)
        side = rng.choice([-1, 1])
        w.add(MovingCircle(x, side * rng.uniform(0.8, 1.6), 0.0, -side * rng.uniform(0.15, 0.6), 0.05))
    elif kind == "emerging":
        # a pedestrian steps out from behind a box that hides it from the LiDAR
        x = rng.uniform(1.5, 3.5)
        side = rng.choice([-1, 1])
        w.add(Box(x, side * 0.45, 0.30, 0.20, 0.0))
        w.add(MovingCircle(x, side * 0.75, 0.0, -side * rng.uniform(0.25, 0.6), 0.05))

    pwm = int(rng.integers(90, 256))
    steer_plan = [(0.0, 0.0)]
    t = 0.0
    while t < 8.0:
        t += rng.uniform(0.8, 2.0)
        steer_plan.append((t, rng.uniform(-0.5, 0.5) if rng.random() < 0.5 else 0.0))

    def driver(t, sim):
        steer = [s for tt, s in steer_plan if tt <= t][-1]
        return steer, pwm if t > 0.3 else 0.0

    scale = rng.uniform(0.75, 1.25)
    return w, driver, pwm, scale, kind


def run_case(seed, adas_on, aeb_config=None):
    rng = np.random.default_rng(seed)
    world, driver, pwm, scale, kind = random_case(rng)
    dyn = Dynamics(speed_model=SpeedModel(v_max=1.0 * scale))
    sim = Simulator(world, (0.0, 0.0, 0.0), adas_on=adas_on, dynamics=dyn, aeb_config=aeb_config,
                    seed=seed)
    sim.run(driver, 8.0, stop_when_stopped=False)
    return sim, pwm, scale, kind


def _both(args):
    seed, aeb_config = args
    off, pwm, scale, kind = run_case(seed, False)
    on, _, _, _ = run_case(seed, True, aeb_config)
    return (seed, pwm, scale, kind, off.car.collided and off.car.impact_speed >= 0.05,
            on.car.collided and on.car.impact_speed >= 0.05, on.car.impact_speed,
            off.car.x, on.car.x, on.t)


def main(runs=200, seed0=1000, aeb_config=None, verbose=True):
    bins = [(90, 130), (130, 170), (170, 210), (210, 256)]
    stats = {b: defaultdict(int) for b in bins}
    by_kind = defaultdict(lambda: defaultdict(int))
    worst = []

    progress = []
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_both, [(seed0 + i, aeb_config) for i in range(runs)], chunksize=2))

    for seed, pwm, scale, kind, off_c, on_c, impact, off_x, on_x, on_t in results:
        b = next(b for b in bins if b[0] <= pwm < b[1])
        stats[b]["n"] += 1
        stats[b]["off"] += off_c
        stats[b]["on"] += on_c
        by_kind[kind]["n"] += 1
        by_kind[kind]["on"] += on_c
        if on_c:
            worst.append((seed, pwm, round(scale, 2), kind, round(impact, 2)))
        if not on_c:
            progress.append(on_x / max(on_t, 1e-6))

    total_on = sum(s["on"] for s in stats.values())
    total_off = sum(s["off"] for s in stats.values())
    if verbose:
        print(f"{runs} random runs  (crashes: ADAS off {total_off}, ADAS on {total_on})")
        print("throttle PWM    runs   crash OFF   crash ON")
        for b, s in stats.items():
            print(f"{b[0]:>4}-{b[1] - 1:<4}     {s['n']:>4}   {s['off']:>8}   {s['on']:>8}")
        print(f"average forward speed of the ADAS-on runs: {np.mean(progress):.2f} m/s")
        print("by pedestrian type:", {k: f"{v['on']}/{v['n']}" for k, v in by_kind.items()})
        if worst:
            print("ADAS-on crashes (seed, pwm, car speed scale, type, impact m/s):")
            for w in worst[:12]:
                print("  ", w)
    return total_on, total_off, worst


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    main(n, s)
