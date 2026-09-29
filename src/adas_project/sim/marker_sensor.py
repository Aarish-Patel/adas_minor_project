"""Simulated marker detector: projects every marker into the car's camera exactly as a
real lens would, applies detection limits and pixel noise, then runs the SAME pose
solver (solvePnP) as the real pipeline. Cheaper than rendering images, and matches
the browser-rendered camera because both use the same geometry.
"""

import math

import numpy as np

from adas.markers import Camera, marker_corners_3d, pose_from_corners, project


class MarkerSensor:
    def __init__(self, camera=None, rate_hz=15.0, pixel_noise=0.6, detect_prob=0.96,
                 min_side_px=18.0, max_view_angle_deg=68.0, seed=0):
        self.cam = camera or Camera()
        self.period = 1.0 / rate_hz
        self.pixel_noise = pixel_noise
        self.detect_prob = detect_prob
        self.min_side_px = min_side_px
        self.max_view_angle = math.radians(max_view_angle_deg)
        self.rng = np.random.default_rng(seed)
        self.next_time = 0.0
        self.observations = []
        self.sizes = {}

    def _visible_through(self, world, p0, p1, ignore_near=0.05):
        """2D line-of-sight from camera to marker against the world's obstacle segments."""
        seg = world.segments()
        if not len(seg):
            return True
        x1, y1 = p0
        x2, y2 = p1
        d = np.array([x2 - x1, y2 - y1])
        L = np.linalg.norm(d)
        if L < 1e-6:
            return True
        ex, ey = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
        den = d[0] * ey - d[1] * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((seg[:, 0] - x1) * ey - (seg[:, 1] - y1) * ex) / den
            u = ((seg[:, 0] - x1) * d[1] - (seg[:, 1] - y1) * d[0]) / den
        hit = (np.abs(den) > 1e-9) & (t > 0.0) & (t < 1.0 - ignore_near / L) & (u >= 0.0) & (u <= 1.0)
        return not hit.any()

    def sense(self, world, car, t):
        """Returns the list of MarkerObs visible now (None if it is not time for a new frame)."""
        if t < self.next_time:
            return None
        self.next_time = t + self.period
        cam = self.cam
        c, s = math.cos(car.theta), math.sin(car.theta)
        obs = []
        for m in world.objects:
            if not hasattr(m, "marker_id"):
                continue
            self.sizes[m.marker_id] = m.size
            n = np.array([math.cos(m.yaw), math.sin(m.yaw)])              # world-frame facing direction
            right = np.array([-n[1], n[0]])
            up = np.array([0.0, 0.0, 1.0])
            local = marker_corners_3d(m.size)
            corners_w = np.array([[m.x + right[0] * px, m.y + right[1] * px, m.z + py] for px, py, _ in local])
            # world -> vehicle frame
            dx, dy = corners_w[:, 0] - car.x, corners_w[:, 1] - car.y
            corners_v = np.column_stack([c * dx + s * dy, -s * dx + c * dy, corners_w[:, 2]])

            centre_v = corners_v.mean(axis=0)
            to_cam = np.array([cam.x - centre_v[0], cam.y - centre_v[1]])
            n_v = np.array([c * n[0] + s * n[1], -s * n[0] + c * n[1]])
            cosang = float(np.dot(n_v, to_cam) / (np.linalg.norm(to_cam) + 1e-9))
            if cosang < math.cos(self.max_view_angle):
                continue                                                  # seen from behind or too oblique

            px, depth = project(cam, corners_v)
            if (depth <= 0.05).any():
                continue
            margin = 4.0
            if (px[:, 0] < margin).any() or (px[:, 0] > cam.width - margin).any() or \
               (px[:, 1] < margin).any() or (px[:, 1] > cam.height - margin).any():
                continue
            side = float(np.linalg.norm(px[1] - px[0]))
            if side < self.min_side_px:
                continue
            cam_w = np.array([car.x + c * cam.x - s * cam.y, car.y + s * cam.x + c * cam.y])
            if not self._visible_through(world, cam_w, (m.x, m.y)):
                continue
            if self.rng.random() > self.detect_prob:
                continue
            noisy = px + self.rng.normal(0.0, self.pixel_noise, px.shape)
            o = pose_from_corners(noisy, m.marker_id, m.size, cam, t)
            if o is not None:
                obs.append(o)
        self.observations = obs
        return obs
