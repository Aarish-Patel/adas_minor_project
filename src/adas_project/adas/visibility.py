"""Scan evidence for remembered points: seen again, seen through, or not visible from here.

A remembered obstacle point is only contradicted by beams that really pass through it: returns within the point's own angular size
(at its range, never less than the LiDAR's beam spacing) that come back from farther away. One miss is weighed by how well the point
could have been seen: from far away a thin object covers less than a beam or two and is easily missed, so the weight falls with
range and with the number of beams expected to hit it; below MIN_BEAMS expected beams the point is not judged at all. A point's width
comes from how many remembered points lie next to it (`width_from_neighbours`): a thin post (one or two points) is judged only close
up, a wall or a box (many points) from farther away. Each store
keeps a clamped score per point (+1 per hit, minus the weight per miss; dropped at zero), the clamped log-odds update of an occupancy
grid (Moravec & Elfes 1985; clamping as in OctoMap, Hornung et al. 2013) with the visibility test of Removert (Kim & Kim 2020).

Incident / user report 14 Oct 2026: thin objects (chair legs) were remembered when close and deleted as soon as the car moved away -
the old rules dropped a point in one scan when no return lay within 10 cm of it, or when anything within +-4 deg was farther.
"""
import numpy as np

HALF_WIDTH_M = 0.01      # a remembered point stands for ~2 cm of surface
SLACK_M = 0.05           # a return within this of the point's range is the point itself
MIN_BEAMS = 4.0          # fewer expected beams on the point than this: not judged from here (a 2 cm post: beyond ~0.9 m)
R_REF_M = 1.0            # misses count fully up to this range, less beyond (R_REF / r)


def beam_step_deg(scan_bearing_deg):
    """The scan's angular spacing (median gap between neighbouring returns), 0.2-2 deg."""
    b = np.sort(np.asarray(scan_bearing_deg, float) % 360.0)
    if len(b) < 10:
        return 1.0
    return float(np.clip(np.median(np.diff(b)), 0.2, 2.0))


def width_from_neighbours(pts, spacing, radius=None, max_width=0.12):
    """Half width (m) of the surface each remembered point stands for: (neighbours within `radius` (3 spacings), itself included,
    minus one) x spacing, at least one spacing, up to max_width; halved. pts: (N, 2) in any metric frame."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return np.empty(0)
    radius = 3.0 * spacing if radius is None else radius
    try:
        from scipy.spatial import cKDTree
        n = np.asarray(cKDTree(pts).query_ball_point(pts, radius, return_length=True), float)
    except ImportError:                                                      # pragma: no cover
        n = np.ones(len(pts))
    return np.clip((n - 1.0) * spacing, spacing, max_width) / 2.0


def width_from_grid(keys, cell, radius_cells=3, max_width=0.12):
    """width_from_neighbours for occupied grid cells (integer keys (N, 2), cell size m): the neighbour count is a box sum over an
    image of the occupied cells (2-D cumulative sums), O(area) instead of a tree search per cell."""
    keys = np.asarray(keys, np.int64).reshape(-1, 2)
    if not len(keys):
        return np.empty(0)
    mn = keys.min(axis=0) - radius_cells
    k = keys - mn
    h, w = k.max(axis=0) + radius_cells + 1
    if h * w > 9_000_000:                                                     # a huge sparse map: fall back to the tree
        return width_from_neighbours(keys * cell, cell, radius=radius_cells * cell * 1.01, max_width=max_width)
    img = np.zeros((h + 1, w + 1), np.int32)
    img[k[:, 0] + 1, k[:, 1] + 1] = 1
    sat = img.cumsum(axis=0).cumsum(axis=1)                                   # summed-area table
    x0, y0 = k[:, 0] - radius_cells, k[:, 1] - radius_cells
    x1, y1 = k[:, 0] + radius_cells + 1, k[:, 1] + radius_cells + 1
    n = sat[x1, y1] - sat[x0, y1] - sat[x1, y0] + sat[x0, y0]
    return np.clip((n - 1.0) * cell, cell, max_width) / 2.0


def evidence(pt_r, pt_b, scan_r, scan_b, min_range=0.0, half_width=HALF_WIDTH_M, slack=SLACK_M, max_beams=48):
    """Remembered points (range pt_r m, bearing pt_b deg, from the LiDAR) against a scan (scan_r, scan_b; valid returns only).
    Returns (hit, miss_weight): hit = a return at the point's range within its angular size; miss_weight in [0, 1] = how strongly the
    scan saw through it (0 = no evidence: occluded by a nearer return, no beams there, or too thin to be seen from here)."""
    pt_r, pt_b = np.asarray(pt_r, float), np.asarray(pt_b, float) % 360.0
    half_width = np.broadcast_to(np.asarray(half_width, float), pt_r.shape)
    n = len(pt_r)
    hit, miss = np.zeros(n, bool), np.zeros(n)
    if not n or not len(scan_r):
        return hit, miss
    step = beam_step_deg(scan_b)
    order = np.argsort(np.asarray(scan_b, float) % 360.0)
    sb = np.asarray(scan_b, float)[order] % 360.0
    sr = np.asarray(scan_r, float)[order]
    sb3, sr3 = np.concatenate([sb - 360.0, sb, sb + 360.0]), np.concatenate([sr, sr, sr])        # wrap-around
    half = np.maximum(np.degrees(np.arctan2(half_width, np.maximum(pt_r, 1e-3))), step / 2.0)
    lo = np.searchsorted(sb3, pt_b - half, side="left")
    hi = np.searchsorted(sb3, pt_b + half, side="right")
    idx = lo[:, None] + np.arange(max_beams)[None, :]
    inside = idx < hi[:, None]
    d = np.where(inside, sr3[np.minimum(idx, len(sr3) - 1)], np.nan)
    with np.errstate(invalid="ignore"):
        at = np.abs(d - pt_r[:, None]) <= slack
        nearer = d < pt_r[:, None] - slack
        farther = d > pt_r[:, None] + slack
    hit = at.any(axis=1)
    n_through = farther.sum(axis=1)
    n_exp = 2.0 * half / step
    n_in = inside.sum(axis=1)                                         # beams examined (at most max_beams)
    judged = ~hit & ~nearer.any(axis=1) & (n_through >= np.maximum(1.0, 0.5 * n_in))
    judged &= (n_exp >= MIN_BEAMS) | (pt_r < min_range)            # inside the minimum range a far return proves the beam passed
    w = np.minimum(1.0, R_REF_M / np.maximum(pt_r, 1e-3)) * np.minimum(1.0, (n_exp - MIN_BEAMS + 1.0) / 4.0)
    w = np.where(pt_r < min_range, 1.0, np.clip(w, 0.0, 1.0))
    miss = np.where(judged, w, 0.0)
    return hit, miss
