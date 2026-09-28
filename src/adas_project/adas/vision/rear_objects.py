"""Objects behind the car from the rear camera, without a trained network: motion parallax and looming.

1. Ground-compensated residual (a form of plane+parallax segmentation, Irani & Anandan 1998): the floor is a plane, so
   between two frames its image moves by a known homography once the car's own motion (dp, dpsi) is known (from the
   wheels / EKF or adas.vision.rear_odometry). Warp the previous frame by that homography; everything lying ON the floor
   cancels, everything standing ABOVE it does not (parallax proportional to its height). The residual blobs are the
   obstacles; the lowest pixel of a blob is where it touches the floor, i.e. a metric position (adas.lane.pixel_to_ground).
2. Looming (Lee 1976, "tau"; used for time-to-collision from a single camera, Horn et al. 2007): the time to contact
   with an approaching object is its image size divided by the growth rate of that size, tau = h / (dh/dt), with no
   need to know its distance or size - fitted by regression over ~0.5 s of a track.
   Works in either direction, so it also shows an object coming at a car that stands still (a person walking up to it).
"""
import collections
import math

import numpy as np

from adas.lane import pixel_to_ground
from adas.markers import project


def ground_homography(cam, dp, dpsi, x_range=(-0.15, -1.2), half_width=0.5):
    """Homography taking pixels of the floor in the previous frame to the current frame, for the car's own motion
    (dp = (dx, dy) in the previous car frame, dpsi = heading change)."""
    import cv2
    q = np.array([[x_range[0], -half_width], [x_range[0], half_width], [x_range[1], half_width], [x_range[1], -half_width]])
    c, s = math.cos(-dpsi), math.sin(-dpsi)
    qn = (q - np.asarray(dp)) @ np.array([[c, s], [-s, c]])                  # R(-dpsi) (q - dp), row vectors
    uv0, _ = project(cam, np.column_stack([q, np.zeros(4)]))
    uv1, _ = project(cam, np.column_stack([qn, np.zeros(4)]))
    return cv2.getPerspectiveTransform(uv0.astype(np.float32), uv1.astype(np.float32))


class Blob:
    __slots__ = ("bbox", "foot", "h", "area", "id")

    def __init__(self, bbox, foot, area):
        self.bbox, self.foot, self.area = bbox, foot, area
        self.h, self.id = float(bbox[3]), None


class RearObjects:
    def __init__(self, cam, diff_thresh=14, min_area=80):
        import cv2
        self.cv2, self.cam, self.thresh, self.min_area = cv2, cam, diff_thresh, min_area

    def detect(self, prev_gray, cur_gray, dp, dpsi):
        """Blobs standing above the floor in the current frame: [Blob(bbox, foot=(x, y) vehicle metres, area)].
        prev_gray may be several frames old (dp, dpsi = the car's motion over that whole interval): parallax grows with
        the baseline, so ~5-8 cm of travel between the two frames finds low or uniform objects a single 2 cm step
        cannot."""
        cv2 = self.cv2
        H = ground_homography(self.cam, dp, dpsi)
        warped = cv2.warpPerspective(prev_gray, H, (self.cam.width, self.cam.height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(np.full_like(prev_gray, 255), H, (self.cam.width, self.cam.height))
        diff = cv2.absdiff(cv2.GaussianBlur(warped, (5, 5), 0), cv2.GaussianBlur(cur_gray, (5, 5), 0))
        mask = ((diff > self.thresh) & (valid > 250)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, stats, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            gx, gy = pixel_to_ground(self.cam, np.array([x + w / 2.0]), np.array([float(y + h)]))
            foot = (float(gx[0]), float(gy[0])) if np.isfinite(gx[0]) else None
            out.append(Blob((int(x), int(y), int(w), int(h)), foot, int(area)))
        return out


class LoomingTracker:
    """Follows blobs from frame to frame and reports time to contact from the growth of their image height."""

    def __init__(self, max_dist_px=40, window_s=0.6, min_samples=5):
        self.tracks = {}
        self.next_id = 1
        self.max_dist, self.window, self.min_samples = max_dist_px, window_s, min_samples

    def update(self, blobs, t):
        """-> [(track id, blob, ttc seconds or None)]; ttc only for objects that are growing (approaching)."""
        used, out = set(), []
        for b in blobs:
            cx, cy = b.bbox[0] + b.bbox[2] / 2.0, b.bbox[1] + b.bbox[3] / 2.0
            best, bd = None, self.max_dist
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - tr["c"][0], cy - tr["c"][1])
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"hist": collections.deque(), "c": (cx, cy), "missed": 0}
            tr = self.tracks[best]
            used.add(best)
            tr["c"], tr["missed"] = (cx, cy), 0
            tr["hist"].append((t, b.h))
            while tr["hist"] and t - tr["hist"][0][0] > self.window:
                tr["hist"].popleft()
            b.id = best
            out.append((best, b, self._ttc(tr)))
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > 5:
                    del self.tracks[tid]
        return out

    def _ttc(self, tr):
        h = tr["hist"]
        if len(h) < self.min_samples:
            return None
        ts = np.array([a for a, _ in h])
        hs = np.array([b for _, b in h])
        if ts[-1] - ts[0] < 0.25:
            return None
        slope = float(np.polyfit(ts, hs, 1)[0])
        if slope <= 1.0:                       # not growing (px/s): not approaching
            return None
        return float(hs[-1] / slope)
