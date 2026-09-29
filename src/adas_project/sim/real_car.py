"""The simulator configured like the REAL car, from what was measured on it.

    from sim.real_car import real_profile
    prof = real_profile()          # prof.params, prof.aeb, prof.speed_model, prof.dynamics, prof.lidar_kw

Sources (all measured on the car with the final wheel grips, pi/tuning_real_car.json + reports):
  * body extents from the LiDAR: front 0.16 m, rear 0.17 m, sides 0.10 m (ruler)          -> vehicle geometry
  * speed model v_max / deadband from sustained LiDAR-measured runs                        -> speed_model
  * stopping: cruise 0.25-0.36 m/s, cut throttle -> rolled 0-7 cm (median 3 cm, 5 runs)    -> brake/coast decel
  * steering: 0.068 rad/m of path curvature per servo degree (turn_test / center_fine)     -> max_inner_*_deg
  * LiDAR: RPLIDAR A3M1 as run on the Pi: ~8 scans/s, ~280 usable points, 0.2 m blind zone
Numbers that are ASSUMPTIONS (not measured) are listed in ASSUMED so reports can say so.
"""
import json
import math
import os
from dataclasses import dataclass, field, replace

from adas.aeb import AEBConfig, SpeedModel
from adas.config import load_tuning
from adas.vehicle_params import VehicleParams

from .car_sim import Dynamics

HERE = os.path.dirname(os.path.abspath(__file__))
TUNING = os.path.join(HERE, "..", "pi", "tuning_real_car.json")
FITTED = os.path.join(HERE, "fitted_car.json")        # written by sim/log_fit.py from real drive logs

K_CURV_PER_SERVO_DEG = 0.068      # rad/m per servo degree (measured)
SERVO_TRAVEL_DEG = (57.0, 53.0)   # servo range right / left of centre (30 .. 87 .. 140)

ASSUMED = ["motor time constant 0.12 s (from the 0.7 s PWM ramp, not fitted)",
           "brake/coast deceleration 2.5 / 4 m/s^2 (fitted to 0-7 cm roll-outs at 0.25-0.36 m/s; speeds above that untested)",
           "servo -> wheel angle ratio 0.78 deg/deg (from curvature per servo degree and wheelbase 0.20 m); measured only up to ~24 servo deg (radius ~0.6 m), full lock (radius ~0.26 m) is an extrapolation"]


@dataclass
class RealProfile:
    params: VehicleParams
    aeb: AEBConfig
    speed_model: object
    dynamics: Dynamics
    lidar_kw: dict = field(default_factory=dict)
    tuning: object = None
    source: str = "hand measurements"


def real_profile(path=TUNING, fitted=None):
    t = load_tuning(path)
    m = t.mount
    # geometry measured from the LiDAR; the simulator wants it from the axles
    lidar_x = 0.12
    front_overhang = m.front_overhang_m - (0.20 - lidar_x)          # LiDAR -> bumper minus LiDAR -> front axle
    rear_overhang = m.rear_overhang_m - lidar_x
    width = m.left_overhang_m + m.right_overhang_m
    ratio = math.degrees(math.atan(K_CURV_PER_SERVO_DEG * t.vehicle.wheelbase))   # wheel deg per servo deg
    params = VehicleParams(wheelbase=t.vehicle.wheelbase, pivot_track=t.vehicle.pivot_track, width=width,
                           front_overhang=front_overhang, rear_overhang=rear_overhang, lidar_x=lidar_x, lidar_y=0.0,
                           max_inner_left_deg=round(ratio * SERVO_TRAVEL_DEG[1], 1),
                           max_inner_right_deg=round(ratio * SERVO_TRAVEL_DEG[0], 1))
    aeb = AEBConfig.for_vehicle(params, t.lidar_min_range)
    for k, v in vars(t.aeb).items():
        if k != "margin":
            setattr(aeb, k, v)
    aeb.margin = max(t.aeb.margin, aeb.margin)
    speed_model, tau, coast = t.speed_model, 0.12, 2.5
    fit = load_fitted(fitted)
    if fit:                                   # values fitted to real drive logs (sim/log_fit.py) beat hand measurements
        sp = fit["speed"]
        speed_model = SpeedModel(v_max=sp["v_max"], deadband=sp["deadband"])
        tau, coast = sp["tau_motor"] + sp.get("delay_s", 0.0), sp["coast_decel"]
        if fit["steer"].get("identifiable"):
            ratio = math.degrees(math.atan(fit["steer"]["k_curv_per_deg"] * t.vehicle.wheelbase))
            params = replace(params, max_inner_left_deg=round(ratio * SERVO_TRAVEL_DEG[1], 1),
                             max_inner_right_deg=round(ratio * SERVO_TRAVEL_DEG[0], 1))
    dyn = Dynamics(speed_model=speed_model, tau_motor=tau, accel_max=3.0, brake_max=4.0, coast_decel=coast, tau_steer=0.08)
    # LiDAR as the car really runs it: Slamtec SDK "Sensitivity" mode (pi/lidar_dense.py), ~1360 samples per
    # rotation (~950 with a return) at ~10 rotations/s, measured on the Pi. With RC_LIDAR_DENSE=0 the car falls
    # back to the rplidar library's standard scan: ~280 samples at ~13 rotations/s - use lidar_legacy for that.
    lidar = dict(rate_hz=10.0, n_points=1360, min_range=t.lidar_min_range, noise_std=0.01, dropout=0.03)
    return RealProfile(params, aeb, speed_model, dyn, lidar, t, "fitted from drive logs" if fit else "hand measurements")


def load_fitted(path=None):
    """The fitted model, or None. Fits of synthetic logs (they carry the true answer) are never used."""
    path = FITTED if path is None else path
    if not path or not os.path.exists(path):
        return None
    try:
        fit = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return None if "truth" in fit else fit


def curvature_to_steer(kappa, params):
    """Path curvature (rad/m, + = left) -> steering command in -1..1 for this vehicle."""
    from adas.vehicle_params import delta_to_steer
    return delta_to_steer(math.atan(kappa * params.wheelbase), params)
