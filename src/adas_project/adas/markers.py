"""ArUco marker perception: camera model, pose from corners, real-image detection.

The same functions serve the simulator and the real Pi:
    pose_from_corners()  corners in pixels -> marker pose in the vehicle frame
    detect_image()       a real camera frame -> list of MarkerObs (OpenCV ArUco)

Vehicle frame: x forward, y left, z up, origin at the rear-axle centre.
Camera frame (OpenCV): x right, y down, z forward.
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Camera:
    """The front camera. The mounting numbers are guesses: measure them on the car."""
    width: int = 640
    height: int = 480
    hfov_deg: float = 68.0
    x: float = 0.24            # m ahead of the rear axle
    y: float = 0.0
    z: float = 0.085           # m above the floor
    pitch_deg: float = 5.0     # tilted down by this much

    @property
    def fx(self):
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.width / 2.0], [0.0, self.fx, self.height / 2.0], [0.0, 0.0, 1.0]])

    def rotation(self):
        """Columns are the camera x/y/z axes expressed in the vehicle frame."""
        p = math.radians(self.pitch_deg)
        return np.array([[0.0, -math.sin(p), math.cos(p)],
                         [-1.0, 0.0, 0.0],
                         [0.0, -math.cos(p), -math.sin(p)]])


@dataclass
class MarkerObs:
    id: int
    corners: np.ndarray                       # (4, 2) pixels: top-left, top-right, bottom-right, bottom-left
    x: float = 0.0                            # marker position in the vehicle frame (m)
    y: float = 0.0
    z: float = 0.0
    facing: float = 0.0                       # direction the marker faces, radians in the vehicle frame
    dist: float = 0.0                         # ground distance from the camera (m)
    bearing: float = 0.0                      # angle to the marker from the car's heading (rad, + left)
    t: float = 0.0
    size_px: float = 0.0


def marker_corners_3d(size):
    h = size / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


def project(cam, points_vehicle):
    """Vehicle-frame points (N,3) -> pixel coordinates (N,2) and depth (N,)."""
    R = cam.rotation()
    origin = np.array([cam.x, cam.y, cam.z])
    pc = (np.asarray(points_vehicle) - origin) @ R          # rows: camera-frame coordinates
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * pc[:, 0] / z + cam.width / 2.0
        v = cam.fx * pc[:, 1] / z + cam.height / 2.0
    return np.column_stack([u, v]), z


def pose_from_corners(corners_px, marker_id, size, cam, t=0.0):
    """Marker pose in the vehicle frame from its four detected corners (solvePnP)."""
    import cv2
    obj = marker_corners_3d(size)
    img = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    Rm, _ = cv2.Rodrigues(rvec)
    R = cam.rotation()
    origin = np.array([cam.x, cam.y, cam.z])
    centre_c = tvec.reshape(3)
    if centre_c[2] <= 0:
        return None
    centre_v = origin + R @ centre_c
    normal_v = R @ (Rm @ np.array([0.0, 0.0, 1.0]))          # marker's outward normal
    facing = math.atan2(normal_v[1], normal_v[0])
    dx, dy = centre_v[0] - cam.x, centre_v[1] - cam.y
    side = np.linalg.norm(img[1] - img[0])
    return MarkerObs(id=marker_id, corners=img, x=float(centre_v[0]), y=float(centre_v[1]), z=float(centre_v[2]),
                     facing=float(facing), dist=float(math.hypot(dx, dy)), bearing=float(math.atan2(dy, dx)),
                     t=t, size_px=float(side))


def detect_image(frame_bgr, cam, sizes, default_size=0.10, dictionary_name="DICT_4X4_50"):
    """Detect ArUco markers in a real camera frame. sizes: {marker_id: side length in metres}."""
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    corners, ids, _ = detector.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        obs = pose_from_corners(c.reshape(4, 2), int(i), sizes.get(int(i), default_size), cam)
        if obs is not None:
            out.append(obs)
    return out
