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
    yaw_deg: float = 0.0       # mounting direction about the vertical: 0 = looks ahead, 180 = looks backwards (rear camera)

    @property
    def fx(self):
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.width / 2.0], [0.0, self.fx, self.height / 2.0], [0.0, 0.0, 1.0]])

    def rotation(self):
        """Columns are the camera x/y/z axes expressed in the vehicle frame."""
        p = math.radians(self.pitch_deg)
        R = np.array([[0.0, -math.sin(p), math.cos(p)],
                      [-1.0, 0.0, 0.0],
                      [0.0, -math.cos(p), -math.sin(p)]])
        if self.yaw_deg:
            c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R      # turn the whole camera about z
        return R


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
    if not np.isfinite(img).all():
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        # IPPE's two solutions coincide when the marker is seen exactly head-on (nose-in parking looks straight at it) and it can return
        # NaN: refine iteratively from a frontal guess (the marker's normal turned toward the camera, depth from its apparent size)
        side = float(np.linalg.norm(img[1] - img[0]))
        if side < 1.0:
            return None
        z0 = cam.fx * size / side
        c = img.mean(axis=0)
        t0 = np.array([[(c[0] - cam.width / 2.0) * z0 / cam.fx], [(c[1] - cam.height / 2.0) * z0 / cam.fx], [z0]])
        ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, None, np.array([[math.pi], [0.0], [0.0]]), t0, True, cv2.SOLVEPNP_ITERATIVE)
        if not ok or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
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


BLUR_TOLERANT = dict(polygonalApproxAccuracyRate=0.08, minMarkerPerimeterRate=0.02, maxErroneousBitsInBorderRate=0.7,
                     errorCorrectionRate=1.0, adaptiveThreshWinSizeMin=5, adaptiveThreshWinSizeMax=65, adaptiveThreshWinSizeStep=15)
# (10 Nov: the fixed-focus webcam blurs a marker 0.5 m away so that the default detector rejects it; with these settings on a 2x upscaled
# frame the same blurred view decodes. Error correction at 1.0 also decodes wrong ids from clutter: use it for ONE known id only and check
# the quad is square - `square_enough`.)


def square_enough(quad, tol=0.40):
    """A decoded quad (4, 2) is a plausible marker: its four sides within +-tol of their mean (a marker seen at an angle foreshortens
    one pair of sides, never by more than this at the angles the car sees it)."""
    q = np.asarray(quad, float).reshape(4, 2)
    s = np.hypot(*(q - np.roll(q, -1, axis=0)).T)
    return bool(s.min() > 4.0 and abs(s.max() - s.mean()) <= tol * s.mean() and abs(s.min() - s.mean()) <= tol * s.mean())


def marker_like(gray, quad, cells=6):
    """Is this quad (4, 2) an ArUco-style marker, whatever its id? The quad is warped to a cells x cells grid (the black border + the inner bits)
    and checked: the border ring mostly dark, the inner bits a mix of dark and light. A blurred far marker is rejected by the decoder (its
    bits are smeared) but still has this structure; a picture frame, a door or a dark box does not (no border/inner contrast). Used by the
    camera service to report 'there is a marker over there' before the id can be read."""
    import cv2
    q = np.asarray(quad, np.float32).reshape(4, 2)
    n = cells * 10
    M = cv2.getPerspectiveTransform(q, np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], np.float32))
    w = cv2.warpPerspective(gray, M, (n, n), flags=cv2.INTER_AREA)
    thr = float(cv2.threshold(w, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    dark = w < thr
    ring = np.ones((n, n), bool)
    ring[10:-10, 10:-10] = False
    inner = dark[10:-10, 10:-10]
    if float(w[~ring].mean() - w[ring].mean()) < 25.0 and float(w.max() - w.min()) < 40:            # no contrast at all
        return False
    return bool(dark[ring].mean() >= 0.80 and 0.15 <= 1.0 - inner.mean() <= 0.85)


def aruco_detector(dictionary_name="DICT_4X4_50", subpixel=False, tolerant=False, with_rejected=False):
    """A function gray image -> (corners, ids) for ArUco markers, for OpenCV 4.7+ (the ArucoDetector class, the laptop) and for the 4.6 of the
    Pi (the module functions). `subpixel`: sub-pixel corner refinement (the PnP of a small marker wants corners better than a pixel)."""
    import cv2
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        if subpixel:
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        for k, v in (BLUR_TOLERANT.items() if tolerant else ()):
            setattr(params, k, v)
        detector = aruco.ArucoDetector(dictionary, params)
        return (lambda gray: detector.detectMarkers(gray)) if with_rejected else (lambda gray: detector.detectMarkers(gray)[:2])
    params = aruco.DetectorParameters_create()
    if subpixel:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    for k, v in (BLUR_TOLERANT.items() if tolerant else ()):
        setattr(params, k, v)
    return ((lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)) if with_rejected
            else (lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)[:2]))


def detect_image(frame_bgr, cam, sizes, default_size=0.10, dictionary_name="DICT_4X4_50"):
    """Detect ArUco markers in a real camera frame. sizes: {marker_id: side length in metres}."""
    import cv2
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    corners, ids = aruco_detector(dictionary_name)(gray)
    out = []
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        obs = pose_from_corners(c.reshape(4, 2), int(i), sizes.get(int(i), default_size), cam)
        if obs is not None:
            out.append(obs)
    return out
