"""Visual odometry from the ground seen by the rear camera: speed and yaw rate without the LiDAR (TODO P20).

The camera looks at the floor behind the car. With the mounting known (height, pitch, yaw; adas.markers.Camera) every
pixel of the floor is a known point of the vehicle frame, so the frame can be resampled into a metric bird's-eye view
(inverse perspective mapping, Bertozzi & Broggi, IEEE T-IP 1998). Between two frames the floor is a rigid body
moving in the opposite way to the car: sparse Lucas-Kanade tracks (Lucas & Kanade 1981) on the bird's-eye views, a robust
similarity fit (RANSAC, cv2.estimateAffinePartial2D) and the geometry below give the car's own motion, in metres, with
a real scale (the camera height) - unlike monocular visual odometry, which has none.

Static floor point in the car frame: q' = R(-dpsi) (q - dp)  ->  fitted R(theta), t: dpsi = -theta, dp = -R(dpsi) t.

Why it is worth having next to the LiDAR (RF2O + EKF speed, 3 cm/s RMSE): an independent measurement of speed and
yaw with a different failure mode (needs texture on the floor, works in a long featureless corridor where LiDAR range
flow is ambiguous), and image motion is affected by vibration differently from the drivetrain model.
"""
import math

import numpy as np

from adas.markers import project


class RearOdometry:
    def __init__(self, cam, x_range=(-0.15, -0.85), y_range=(-0.35, 0.35), res=0.005, max_corners=150):
        import cv2
        self.cv2, self.cam, self.res, self.max_corners = cv2, cam, res, max_corners
        rows = int(abs(x_range[1] - x_range[0]) / res)
        cols = int((y_range[1] - y_range[0]) / res)
        r, c = np.mgrid[0:rows, 0:cols]
        self.gx = x_range[0] + np.sign(x_range[1] - x_range[0]) * r * res          # rows run away from the car
        self.gy = y_range[1] - c * res                                              # columns run right-to-left in y
        uv, z = project(cam, np.column_stack([self.gx.ravel(), self.gy.ravel(), np.zeros(rows * cols)]))
        self.mapx = uv[:, 0].reshape(rows, cols).astype(np.float32)
        self.mapy = uv[:, 1].reshape(rows, cols).astype(np.float32)
        self.valid = (z.reshape(rows, cols) > 0.02) & (self.mapx >= 0) & (self.mapx < cam.width) & \
                     (self.mapy >= 0) & (self.mapy < cam.height)
        # never pick features on the edge of the valid region: that edge is fixed in the bird's-eye view and would pull
        # the fit toward 'no motion'
        self.feature_mask = cv2.erode(self.valid.astype(np.uint8) * 255, np.ones((31, 31), np.uint8))
        self.prev = None
        self.prev_t = None
        self.rows, self.cols = rows, cols
        self.x_sign = float(np.sign(x_range[1] - x_range[0]))

    def bev(self, gray):
        out = self.cv2.remap(gray, self.mapx, self.mapy, self.cv2.INTER_LINEAR, borderMode=self.cv2.BORDER_CONSTANT)
        out[~self.valid] = 0
        return out

    def _to_vehicle(self, pts):
        """BEV pixel coordinates (col, row) -> vehicle-frame metres."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        return np.column_stack([self.gx[0, 0] + self.x_sign * pts[:, 1] * self.res, self.gy[0, 0] - pts[:, 0] * self.res])

    def update(self, frame, t):
        """One frame (BGR or gray) at time t -> (dx, dy, dpsi, quality) of the car since the last frame, or None."""
        cv2 = self.cv2
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b = self.bev(gray)
        prev, prev_t = self.prev, self.prev_t
        self.prev, self.prev_t = b, t
        if prev is None or t - prev_t < 1e-3:
            return None
        p0 = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 6, mask=self.feature_mask)
        if p0 is None or len(p0) < 8:
            return None
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, b, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 8:
            return None
        q0, q1 = self._to_vehicle(p0[ok]), self._to_vehicle(p1[ok])
        M, inl = cv2.estimateAffinePartial2D(q0.astype(np.float32), q1.astype(np.float32), method=cv2.RANSAC,
                                             ransacReprojThreshold=0.006)
        if M is None:
            return None
        theta = math.atan2(M[1, 0], M[0, 0])
        tvec = M[:, 2]
        dpsi = -theta
        c, s = math.cos(dpsi), math.sin(dpsi)
        dp = -np.array([[c, -s], [s, c]]) @ tvec
        quality = float(inl.sum()) / max(1, len(q0))
        return float(dp[0]), float(dp[1]), float(dpsi), quality

    def speed_yaw(self, frame, t):
        """(v m/s along the car's heading, yaw rate rad/s, quality) from consecutive frames, or None."""
        dt_prev = self.prev_t
        r = self.update(frame, t)
        if r is None or dt_prev is None:
            return None
        dt = t - dt_prev
        return r[0] / dt, r[2] / dt, r[3]
