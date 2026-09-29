"""Auto-park (TODO F1): find a perpendicular bay beside the car from the LiDAR and back into it.

Detection works on the scan alone (no map, no markings): along one side of the car the parked objects form a row; a gap in that
row that is wide enough for the body plus clearance, bounded by objects on both ends and free to the back of the bay, is a bay.
This is the free-space-slot idea of ultrasonic / LiDAR parking-slot detection (e.g. Suhr & Jung, "Fully-automatic recognition of
various parking slot markings using a hierarchical tree structure", Optical Engineering 2013, for the camera version; Jeong et al.,
"Parking space detection from LiDAR", for the range version). Driving in is the existing pose-goal planner (Hybrid A* with
Dubins / reversing legs, adas/autonav.py): the goal is the bay centre with the nose pointing out (reverse-in parking, what
production valet-parking does so the car can leave forward).

    bays = find_bays(points_vehicle)                       # nearest first
    goal = park_goal(bays[0])                              # (x, y, heading_deg) in the vehicle frame now
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Bay:
    side: int            # +1 left, -1 right of the car
    x: float             # the bay's centre along the car's heading (vehicle frame, m)
    y_entry: float       # lateral position of the row's edge (the mouth of the bay)
    width: float         # gap between the neighbours (m)
    depth: float         # free depth measured from the mouth (m)


def find_bays(pts, side=None, car_width=0.20, clearance=0.06, min_depth=0.40, x_range=(-0.4, 3.0), row=(0.30, 1.0), res=0.02):
    """Bays on the left (+1), right (-1) or both sides (None), nearest first. pts: LiDAR returns in the vehicle frame (N, 2)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    out = []
    for sd in ((1, -1) if side is None else (side,)):
        lat = pts[:, 1] * sd
        n = int((x_range[1] - x_range[0]) / res)
        occ = np.zeros(n, bool)
        band = pts[(lat > row[0]) & (lat < row[1]) & (pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1])]
        if len(band) == 0:
            continue
        idx = ((band[:, 0] - x_range[0]) / res).astype(int)
        occ[np.clip(idx, 0, n - 1)] = True
        k = int(0.06 / res)                                   # bridge the gaps between LiDAR points on one object
        occ = np.convolve(occ.astype(int), np.ones(2 * k + 1, int), "same") > 0
        # runs of free x between two occupied runs
        i = 0
        while i < n:
            if occ[i]:
                i += 1
                continue
            j = i
            while j < n and not occ[j]:
                j += 1
            left_obj, right_obj = i > 0 and occ[i - 1], j < n and occ[j]
            width = (j - i) * res
            if left_obj and right_obj and car_width + 2 * clearance <= width <= car_width + 0.5:
                x0, x1 = x_range[0] + i * res, x_range[0] + j * res
                # depth: free of returns from the mouth to the back of the bay (the back wall, if seen, ends it)
                inside = pts[(pts[:, 0] > x0 + 0.03) & (pts[:, 0] < x1 - 0.03) & (lat > row[0] - 0.05)]
                ly = np.sort(inside[:, 1] * sd) if len(inside) else np.array([])
                y_entry = float(np.median(band[:, 1] * sd)) if len(band) else row[0]
                y_entry = float(np.min(band[(band[:, 0] < x0 - 0.02) | (band[:, 0] > x1 + 0.02), 1] * sd)) \
                    if np.any((band[:, 0] < x0 - 0.02) | (band[:, 0] > x1 + 0.02)) else y_entry
                back = float(ly[0]) if len(ly) else y_entry + 1.2
                depth = back - y_entry
                if depth >= min_depth:
                    out.append(Bay(sd, 0.5 * (x0 + x1), sd * y_entry, width, depth))
            i = j
    return sorted(out, key=lambda b: abs(b.x - 0.1))


def park_goal(bay, car_length=0.36, rear_axle_to_centre=0.115, margin=0.03):
    """Goal pose for the REAR AXLE (x, y, heading deg, + = left) that backs the car into the bay, nose pointing out."""
    depth_target = min(bay.depth - 0.05, car_length + 0.06)          # how far in the car centre should go
    yc = bay.y_entry + bay.side * (0.5 * depth_target)
    heading = -bay.side * 90.0                                       # nose points out of the bay
    # the rear axle is `rear_axle_to_centre` behind the body centre along the heading: nose out => axle is deeper in
    ya = yc + bay.side * rear_axle_to_centre
    return (bay.x, ya, heading)
