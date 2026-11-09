"""The parking camera's set-up: where it is mounted, which marker it looks for, and the camera model built from it.

`pi/camera.json` (edited by hand on the car; defaults here) holds:
    facing         "front" (drive nose in, marker on the bay's wall ahead) or "rear" (reverse in)
    marker_id      ArUco id (dictionary DICT_4X4_50)       marker_size_m  side of the black square, metres
    x, z, pitch_deg, hfov_deg   mounting: metres ahead of the rear axle, height, tilt down, horizontal field of view (optional: the
                   defaults of adas.park_select.front_camera / rear_camera)
    index, width, height, fps   the webcam for the camera service (pi/marker_camera.py)
    rotate_deg     0, or 180 for a camera mounted upside down (the service turns the frame upright before detection and streaming)
A calibration (tools/camera_calibrate.py, only when asked) writes the same keys plus fx, cx ... to `adas/camera_<facing>.json`; its
mounting numbers and field of view then replace the guessed ones.
"""
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG_PATH = os.path.join(ROOT, "pi", "camera.json")
DEFAULTS = {"facing": "front", "marker_id": 23, "marker_size_m": 0.08, "index": 0, "width": 640, "height": 480, "fps": 15, "rotate_deg": 0}
MOUNT_KEYS = ("x", "y", "z", "pitch_deg", "yaw_deg", "hfov_deg")


def load(path=None):
    """The config dict (defaults for what the file does not say). `path`, else $RC_CAMERA_CONFIG, else pi/camera.json."""
    cfg = dict(DEFAULTS)
    try:
        with open(path or os.environ.get("RC_CAMERA_CONFIG") or CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    if cfg["facing"] not in ("front", "rear"):
        raise ValueError(f"camera.json: facing must be 'front' or 'rear', not {cfg['facing']!r}")
    return cfg


def camera_model(cfg=None):
    """The adas.markers.Camera for the config: the preset of its facing, then the calibration file's numbers, then the config's own."""
    from dataclasses import replace
    from .park_select import camera_for
    cfg = cfg or load()
    cam = replace(camera_for(cfg["facing"]), width=int(cfg["width"]), height=int(cfg["height"]))
    try:
        with open(os.path.join(ROOT, "adas", f"camera_{cfg['facing']}.json"), encoding="utf-8") as f:
            cal = json.load(f)
        fields = {k: cal[k] for k in MOUNT_KEYS if k in cal}
        if "fx" in cal and "width" in cal:                                   # a calibrated field of view at the calibration's size
            fields["hfov_deg"] = math.degrees(2 * math.atan(cal["width"] / 2 / cal["fx"]))
        cam = replace(cam, **fields)
    except (OSError, ValueError, KeyError):
        pass
    return replace(cam, **{k: float(cfg[k]) for k in MOUNT_KEYS if k in cfg})
