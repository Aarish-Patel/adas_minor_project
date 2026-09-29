"""Robustness sweeps: how does the ADAS cope when the real car differs from what it assumes?

    python -m sim.sweeps [runs_per_point]

Each sweep varies ONE thing, runs the same random drives (ADAS off vs on), and reports
crash rates. This is what to look at before tuning on the real car: a flat line means
the parameter does not matter; a steep one is worth calibrating carefully.
"""

import dataclasses
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from adas.aeb import AEBConfig, SpeedModel
from adas.vehicle_params import VehicleParams

from .car_sim import Dynamics
from .intent_data import MODEL_DIR
from .lidar_sim import LidarSim
from .monte_carlo import random_case
from .simulator import Simulator

SWEEPS = {
    "speed_calibration": ("Real speed / calibrated speed", [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]),
    "link_delay": ("Link delay (s)", [0.0, 0.03, 0.08, 0.15, 0.25]),
    "lidar_noise": ("LiDAR noise (m)", [0.0, 0.01, 0.02, 0.04]),
    "lidar_dropout": ("LiDAR dropouts", [0.0, 0.05, 0.15, 0.3]),
    "braking": ("Real braking (m/s2)", [0.8, 1.2, 2.5, 4.0]),
}


def _run(args):
    sweep, value, seed, adas_on = args
    rng = np.random.default_rng(seed)
    world, driver, pwm, _, kind = random_case(rng)
    dyn = Dynamics()
    lidar = LidarSim(seed=seed)
    delay = None
    if sweep == "speed_calibration":
        dyn = Dynamics(speed_model=SpeedModel(v_max=1.0 * value))
    elif sweep == "link_delay":
        delay = value
    elif sweep == "lidar_noise":
        lidar.noise_std = value
    elif sweep == "lidar_dropout":
        lidar.dropout = value
    elif sweep == "braking":
        dyn = Dynamics(brake_max=value, coast_decel=min(1.5, value))
    sim = Simulator(world, (0.0, 0.0, 0.0), adas_on=adas_on, dynamics=dyn, lidar=lidar, seed=seed)
    if delay is not None:
        sim.link_delay = delay
    sim.run(driver, 8.0, stop_when_stopped=False)
    return sim.car.collided and sim.car.impact_speed >= 0.05


def main(runs=60):
    out = {}
    with ProcessPoolExecutor() as pool:
        for name, (label, values) in SWEEPS.items():
            rows = []
            for v in values:
                seeds = range(3000, 3000 + runs)
                off = list(pool.map(_run, [(name, v, s, False) for s in seeds], chunksize=4))
                on = list(pool.map(_run, [(name, v, s, True) for s in seeds], chunksize=4))
                rows.append({"value": v, "off": 100.0 * sum(off) / runs, "on": 100.0 * sum(on) / runs})
                print(f"{name:18s} {v:6}:  crash rate  ADAS off {rows[-1]['off']:5.1f} %   ADAS on {rows[-1]['on']:5.1f} %")
            out[name] = {"label": label, "rows": rows}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "sweeps.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
