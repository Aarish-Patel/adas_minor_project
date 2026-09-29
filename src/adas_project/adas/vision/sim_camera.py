"""A synthetic camera for testing the vision code without hardware: renders what a camera mounted on the car (any
adas.markers.Camera - forward or rear) would see of a textured floor, boxes and ArUco markers, from a vehicle pose.

The floor is a random blurred texture in world coordinates (features for optical flow, like carpet or a table top),
looked up per pixel through the same camera geometry the vision code inverts (adas.lane.pixel_to_ground), so the
ground truth is exact. Boxes are drawn as shaded faces (painter's algorithm), markers by warping cv2.aruco images.
"""
import math

import numpy as np

from adas.lane import pixel_to_ground
from adas.markers import project


def world_to_vehicle(pts, pose):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    d = np.asarray(pts, float) - np.array([x, y, 0.0])
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]])


class SimCamera:
    def __init__(self, cam, size=(320, 240), seed=0, tile_m=8.0, res=0.01, noise=2.0):
        import cv2
        self.cv2, self.cam = cv2, cam
        self.w, self.h = size
        self.cam = type(cam)(**{**cam.__dict__, "width": self.w, "height": self.h})    # render at this resolution
        rng = np.random.default_rng(seed)
        n = int(tile_m / res)
        tex = rng.random((n, n)).astype(np.float32)
        big = np.tile(tex, (3, 3))                              # blur a 3x3 tiling and keep the middle tile: seamless
        big = cv2.GaussianBlur(big, (0, 0), 3.0) * 0.6 + cv2.GaussianBlur(big, (0, 0), 9.0) * 0.4
        tex = big[n:2 * n, n:2 * n]
        tex = (tex - tex.min()) / (tex.max() - tex.min())
        fine = cv2.GaussianBlur(np.tile(rng.random((n, n)).astype(np.float32), (3, 3)), (0, 0), 1.2)[n:2 * n, n:2 * n]
        tex = 0.7 * tex + 0.3 * (fine - fine.min()) / (fine.max() - fine.min())     # coarse blobs + fine grain
        self.tex, self.tile, self.res, self.noise = (tex * 200 + 30).astype(np.uint8), tile_m, res, noise
        u, v = np.meshgrid(np.arange(self.w, dtype=np.float32) + 0.5, np.arange(self.h, dtype=np.float32) + 0.5)
        gx, gy = pixel_to_ground(self.cam, u.ravel(), v.ravel())
        self.gx = gx.reshape(u.shape).astype(np.float32)
        self.gy = gy.reshape(u.shape).astype(np.float32)
        self.sky = np.isnan(self.gx)
        self.rng = rng

    def render(self, pose, boxes=(), markers=(), board=None):
        """pose: (x, y, heading) of the vehicle in the world. boxes: [(cx, cy, length, width, height, heading)].
        markers: [(id, cx, cy, cz, facing_rad, size)] standing markers. Returns a BGR uint8 frame."""
        cv2 = self.cv2
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        wx = x + c * self.gx - s * self.gy
        wy = y + s * self.gx + c * self.gy
        mx = np.mod(np.nan_to_num(wx) / self.res, self.tex.shape[1]).astype(np.float32)
        my = np.mod(np.nan_to_num(wy) / self.res, self.tex.shape[0]).astype(np.float32)
        gray = cv2.remap(self.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        if board is not None:                                   # a checkerboard lying on the floor (calibration target)
            bx, by, cols, rows, sq = board
            u_, v_ = (wx - bx) / sq, (wy - by) / sq
            inside = (u_ >= 0) & (u_ < cols) & (v_ >= 0) & (v_ < rows) & ~self.sky
            chk = (np.floor(u_) + np.floor(v_)) % 2 == 0
            gray[inside] = np.where(chk[inside], 235, 25).astype(np.uint8)
            m = 0.6 * sq                                         # a white margin so the pattern is detectable
            edge = (u_ >= -m / sq) & (u_ < cols + m / sq) & (v_ >= -m / sq) & (v_ < rows + m / sq) & ~inside & ~self.sky
            gray[edge] = 235
        gray[self.sky] = 210
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # boxes: far to near
        items = []
        for (cx, cy, L, W, H, hd) in boxes:
            ch, sh = math.cos(hd), math.sin(hd)
            base = np.array([[L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2], [L / 2, -W / 2]])
            wxy = np.column_stack([cx + ch * base[:, 0] - sh * base[:, 1], cy + sh * base[:, 0] + ch * base[:, 1]])
            corners = np.vstack([np.column_stack([wxy, np.zeros(4)]), np.column_stack([wxy, np.full(4, H)])])
            pv = world_to_vehicle(corners, pose)
            uv, z = project(self.cam, pv)
            if (z <= 0.02).any():
                continue
            items.append((float(np.mean(z)), uv, z))
        for depth, uv, z in sorted(items, key=lambda t: -t[0]):
            faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
            shade = [150, 110, 130, 90, 175]
            order = sorted(range(5), key=lambda i: -np.mean(z[list(faces[i])]))
            for i in order:
                cv2.fillConvexPoly(frame, uv[list(faces[i])].astype(np.int32), (shade[i], shade[i], shade[i] + 30))
        for (mid, cx, cy, cz, facing, size) in markers:
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            img = cv2.aruco.generateImageMarker(d, mid, 200)
            img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            n = np.array([math.cos(facing), math.sin(facing), 0.0])
            right = np.array([-math.sin(facing), math.cos(facing), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            hs = size / 2 * (240 / 200)
            c3 = np.array([cx, cy, cz])
            quad = np.array([c3 - hs * right + hs * up, c3 + hs * right + hs * up, c3 + hs * right - hs * up,
                             c3 - hs * right - hs * up])
            uv, z = project(self.cam, world_to_vehicle(quad, pose))
            if (z <= 0.02).any():
                continue
            H = cv2.getPerspectiveTransform(np.float32([[0, 0], [239, 0], [239, 239], [0, 239]]), uv.astype(np.float32))
            warped = cv2.warpPerspective(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), H, (self.w, self.h))
            mask = cv2.warpPerspective(np.full((240, 240), 255, np.uint8), H, (self.w, self.h))
            frame[mask > 0] = warped[mask > 0]
        if self.noise:
            frame = np.clip(frame.astype(np.float32) + self.rng.normal(0, self.noise, frame.shape), 0, 255).astype(np.uint8)
        return frame
