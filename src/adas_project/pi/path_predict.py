"""Steering-aware predicted-path stopping distance, using tonight's REAL measured turn
radii (M11's converged turn_table) instead of the theoretical bicycle-model angle from
vehicle_params.py's steer_to_delta(), which has never been independently checked against
real hardware. Reuses adas/geometry.py's already-tested arc-sweep footprint math (full
vehicle footprint swept along the arc, not just a cone) by picking the `delta` value that
reproduces the MEASURED radius, rather than adas/geometry's own theoretical delta mapping.

Measured turn_table (pi/legacy/reports/steering_diag_report_v7.json, converged, strict pass-to-pass
stability): offset -35deg -> 1.351m, +20deg -> 2.184m, +35deg -> 0.633m. The -20deg point
was flagged unreliable (a near-zero outlier breaking an otherwise clean monotonic pattern)
and is excluded here. Left/right radii at +-35deg differ by more than 2x (1.351 vs 0.633) -
this could be a real mechanical asymmetry or still just noise (this session repeatedly saw
the asymmetry ratio change across runs); since this feeds a safety-critical stop decision,
each side uses ONLY its own measured data rather than assuming symmetry, which is the more
conservative choice when the two sides disagree.
"""
import math

import numpy as np

from adas.geometry import travel_distance_to_contact
from adas.vehicle_params import VehicleParams

# (offset_magnitude_deg, radius_m) - positive-offset (steer_a raw > center) side
_POS_TABLE = [(0.0, 1e6), (20.0, 2.184), (35.0, 0.633)]
# negative-offset side - -20deg excluded (unreliable outlier), only 0 and -35 trustworthy
_NEG_TABLE = [(0.0, 1e6), (35.0, 1.351)]

def _measured():
    """The measured body and fitted steering the path gate uses (pi/relay_assists.car_params, the tuning file's
    ruler measurements; 0.0656 rad/m per servo degree from the logging drive). Falls back to the old numbers if the
    tuning file is missing."""
    try:
        import os
        from adas.config import load_tuning
        from pi.relay_assists import K_CURV_PER_SERVO_DEG, car_params
        tun = load_tuning(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_real_car.json"))
        return car_params(tun.mount), K_CURV_PER_SERVO_DEG, float(tun.servo.left_center)
    except Exception:
        return VehicleParams(wheelbase=0.20, pivot_track=0.07, width=0.14, front_overhang=0.06, rear_overhang=0.05,
                             lidar_x=0.12, lidar_y=0.0), None, 90.0


VP, K_FIT, SERVO_CENTRE = _measured()


def radius_for_offset(offset_deg):
    """Measured turn radius (m, always positive) for a raw steering offset from center."""
    table = _POS_TABLE if offset_deg >= 0 else _NEG_TABLE
    mags = [t[0] for t in table]
    radii = [t[1] for t in table]
    return float(np.interp(abs(offset_deg), mags, radii))


def delta_for_offset(offset_deg):
    """The `delta` (bicycle-model steering angle, radians) that reproduces the MEASURED
    radius via adas.geometry's kappa=tan(delta)/wheelbase convention - a back-calculated
    value, not vehicle_params.steer_to_delta()'s theoretical one.
    offset_deg = servo - 90; with the fitted model the straight-ahead servo is the calibrated centre, and a
    positive result turns toward +y of this module's frame (the right, as the relay's legacy code uses it)."""
    if K_FIT is not None:
        return math.atan(K_FIT * (offset_deg + 90.0 - SERVO_CENTRE) * VP.wheelbase)
    if abs(offset_deg) < 0.5:
        return 0.0
    R = radius_for_offset(offset_deg)
    kappa = 1.0 / R
    if offset_deg < 0:
        kappa = -kappa
    return math.atan(kappa * VP.wheelbase)


def points_to_vehicle_frame(car_frame_points):
    """car_frame_points: list of (bearing_deg, dist_m), LiDAR-relative (0deg = car's front,
    LiDAR at its own mount position). Converts to (x, y) in adas.geometry's vehicle frame
    (origin at rear-axle centre, x forward, y left) by rotating to cartesian then shifting
    by the LiDAR's own offset from the rear axle."""
    if not car_frame_points:
        return np.zeros((0, 2))
    arr = np.array(car_frame_points, dtype=float)
    rad = np.radians(arr[:, 0])
    dist = arr[:, 1]
    x = dist * np.cos(rad) + VP.lidar_x
    y = dist * np.sin(rad) + VP.lidar_y
    return np.stack([x, y], axis=1)


def predicted_stop_distance(car_frame_points, offset_deg, direction, horizon=2.5):
    """Distance (m, along the arc) the car can travel at this steering offset before its
    swept footprint would touch something - using the REAL measured turn radius, not a
    fixed cone. direction: +1 forward, -1 reverse."""
    pts = points_to_vehicle_frame(car_frame_points)
    delta = delta_for_offset(offset_deg)
    return travel_distance_to_contact(pts, delta, direction, VP, horizon=horizon)
