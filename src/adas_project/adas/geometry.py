"""Predicted-path geometry: where would the car go, and what would it hit."""

import numpy as np

from .vehicle_params import VehicleParams


def path_pose(delta, s, p):
    """Rear-axle pose after travelling signed arc length s at steering angle delta.

    Returns (x, y, psi) in the current vehicle frame. s may be an array.
    """
    kappa = np.tan(delta) / p.wheelbase
    s = np.asarray(s, dtype=float)
    psi = kappa * s
    if abs(kappa) < 1e-6:
        return s, np.zeros_like(s), psi
    return np.sin(psi) / kappa, (1.0 - np.cos(psi)) / kappa, psi


def _corridor_filter(pts, delta, direction, p, horizon, margin):
    """Keep only points that can possibly be swept by the footprint (cheap analytic test)."""
    half_w = p.width / 2.0 + margin
    kappa = np.tan(delta) / p.wheelbase
    if abs(kappa) < 1e-4:
        along = pts[:, 0] * direction
        keep = ((np.abs(pts[:, 1]) <= half_w) & (along <= horizon + max(p.front_x, -p.rear_x) + margin)
                & (along >= -max(p.front_x, -p.rear_x) - margin))
        return pts[keep]
    R = 1.0 / kappa
    rho = np.hypot(pts[:, 0], pts[:, 1] - R)
    r_in = abs(R) - half_w
    r_out = np.hypot(abs(R) + half_w, max(p.front_x, -p.rear_x) + margin)
    keep = (rho >= r_in - 1e-3) & (rho <= r_out + 1e-3)
    return pts[keep]


def travel_distance_to_contact(points, delta, direction, p, horizon=2.0, step=0.02, margin=0.02):
    """How far the car can travel along its current steering arc before touching a point.

    points:    (N, 2) obstacle points in the vehicle frame
    delta:     bicycle steering angle (rad)
    direction: +1 forward, -1 reverse
    margin:    extra clearance added around the footprint (m)

    The whole footprint (not just the bumper) is swept along the arc, so a wall
    beside the car on a tight turn is caught correctly. Returns inf if nothing
    is hit within `horizon`. Distance is measured for the rear-axle centre.
    """
    if len(points) == 0:
        return np.inf

    reach = horizon + p.front_x + margin + 0.1
    pts = points[np.hypot(points[:, 0], points[:, 1]) < reach]
    if len(pts) == 0:
        return np.inf
    pts = _corridor_filter(pts, delta, direction, p, horizon, margin)
    if len(pts) == 0:
        return np.inf

    s_vals = direction * np.arange(0.0, horizon + step, step)
    px, py, psi = path_pose(delta, s_vals, p)

    dx = pts[None, :, 0] - px[:, None]
    dy = pts[None, :, 1] - py[:, None]
    cos_p = np.cos(psi)[:, None]
    sin_p = np.sin(psi)[:, None]
    local_x = cos_p * dx + sin_p * dy
    local_y = -sin_p * dx + cos_p * dy

    half_w = p.width / 2.0 + margin
    inside = ((local_x >= p.rear_x - margin) & (local_x <= p.front_x + margin) &
              (np.abs(local_y) <= half_w))

    hit_steps = np.flatnonzero(inside.any(axis=1))
    if len(hit_steps) == 0:
        return np.inf
    return float(abs(s_vals[hit_steps[0]]))
