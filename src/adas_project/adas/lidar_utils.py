"""LiDAR scan -> obstacle points in the vehicle frame."""

import numpy as np

from .vehicle_params import VehicleParams


def scan_to_points(angles, ranges, valid, p):
    """angles: radians, measured counter-clockwise from the vehicle's forward (+x) axis
    (a real RPLIDAR reports clockwise angles; convert in the driver, not here).
    Returns (N, 2) points relative to the rear-axle centre.
    """
    a = angles[valid]
    r = ranges[valid]
    x = p.lidar_x + r * np.cos(a)
    y = p.lidar_y + r * np.sin(a)
    return np.column_stack([x, y])
