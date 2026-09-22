"""One running simulation, driven in real time, exposing its state for the viewer.

Network-free on purpose: server.py wraps this, and tests can use it directly.
"""

import dataclasses
import json
import math
import os
import time

import numpy as np

from adas.adaptive import AdaptiveWarning, warning_row
from adas.aeb import AEBConfig, SpeedModel
from adas.geometry import path_pose, travel_distance_to_contact
from adas.intent import IntentEstimator, IntentLogger, IntentModel
from adas.vehicle_params import VehicleParams, steer_to_delta, steer_to_wheel_angles
from adas.warning import RiskScorer

from .car_sim import Dynamics
from .drivers import HumanLikeDriver
from .intent_data import MODEL_DIR
from .library import SCENARIOS
from .simulator import DT, Simulator

# (id, label, group, min, max, step, unit, help)
TUNABLES = [
    ("aeb.decel", "Assumed braking deceleration", "ADAS - braking", 0.5, 3.0, 0.05, "m/s²",
     "How hard the ADAS believes the car can brake. Lower = brakes earlier (safer)."),
    ("aeb.latency", "System latency", "ADAS - braking", 0.05, 0.4, 0.01, "s",
     "Scan age + link + processing delay added to the stopping distance."),
    ("aeb.margin", "Stop margin", "ADAS - braking", 0.02, 0.25, 0.005, "m",
     "Gap to keep in front of an obstacle. Must cover the LiDAR blind zone."),
    ("aeb.brake_pwm_max", "Strongest active brake", "ADAS - braking", 40, 255, 5, "PWM",
     "Reverse-motor command used to brake hard."),
    ("aeb.max_pwm", "Top-speed cap", "ADAS - speed scaling", 100, 255, 5, "PWM",
     "The driver's throttle is limited to this PWM."),
    ("aeb.aware_margin", "Awareness width", "ADAS - speed scaling", 0.0, 0.45, 0.01, "m",
     "Obstacles this close to either side of the path slow the car down."),
    ("aeb.aware_decel", "Awareness slow-down", "ADAS - speed scaling", 0.3, 2.5, 0.05, "m/s²",
     "How gently speed is scaled down near side obstacles."),
    ("aeb.v_floor", "Awareness speed floor", "ADAS - speed scaling", 0.1, 0.9, 0.02, "m/s",
     "Awareness never slows the car below this on its own."),
    ("model.v_max", "Speed calibration (PWM 255)", "ADAS - speed model", 0.4, 2.0, 0.02, "m/s",
     "What the ADAS believes the top speed is. Calibrate on the real car."),
    ("model.deadband", "Motor dead-band", "ADAS - speed model", 0, 100, 1, "PWM",
     "PWM below this does not move the car."),
    ("truth.v_scale", "Real speed vs calibration", "Reality (simulator only)", 0.6, 1.5, 0.01, "x",
     "1.0 = the ADAS calibration is perfect. Try 1.25 to test robustness."),
    ("truth.tau_motor", "Motor response time", "Reality (simulator only)", 0.05, 0.5, 0.01, "s", ""),
    ("truth.brake_max", "Real braking", "Reality (simulator only)", 0.8, 5.0, 0.1, "m/s²", ""),
    ("truth.coast_decel", "Coasting deceleration", "Reality (simulator only)", 0.3, 3.0, 0.1, "m/s²", ""),
    ("sim.link_delay", "Link delay", "Reality (simulator only)", 0.0, 0.25, 0.005, "s",
     "Delay between the Pi's command and the car acting on it."),
    ("lidar.noise_std", "LiDAR noise", "Reality (simulator only)", 0.0, 0.05, 0.001, "m", ""),
    ("lidar.dropout", "LiDAR dropouts", "Reality (simulator only)", 0.0, 0.3, 0.005, "", ""),
    ("driver.lapse_rate", "Virtual driver: lapses", "Virtual driver", 0.0, 0.6, 0.01, "/s",
     "How often the virtual driver looks away."),
    ("driver.reaction", "Virtual driver: reaction time", "Virtual driver", 0.05, 0.8, 0.01, "s", ""),
]
TUNABLE_IDS = {t[0] for t in TUNABLES}


class Session:
    def __init__(self):
        self.p = VehicleParams()
        self.speed = 1.0
        self.paused = False
        self.mode = "active"
        self.input = (0.0, 0.0)
        self.events = []
        self.tune_overrides = {}
        self.cv_dets = []
        self.cv_time = 0.0
        self.recording = None
        self.intent_model = None
        self.adaptive = None
        path = os.path.join(MODEL_DIR, "intent.joblib")
        if os.path.exists(path):
            self.intent_model = IntentModel.load(path)
        apath = os.path.join(MODEL_DIR, "adaptive_warning.joblib")
        if AdaptiveWarning.available(apath):
            try:
                from adas.adaptive import COLUMNS
                candidate = AdaptiveWarning(apath)
                # a model trained with a different feature set cannot be used: fall back to physics risk only
                self.adaptive = candidate if list(candidate.columns) == list(COLUMNS) else None
            except Exception:
                self.adaptive = None
        self.scorer = RiskScorer(self.p)
        self.load("playground")

    # ------------------------------------------------------------------ setup
    def load(self, scenario_id, seed=None):
        spec = SCENARIOS[scenario_id]
        self.scenario_id = scenario_id
        self.spec = spec
        build = spec["build"]
        world, start = build(seed) if (seed is not None and scenario_id == "virtual") else build()
        self.world = world
        self.start = start
        self.driver_kind = spec["driver"]

        aeb_cfg = AEBConfig.for_vehicle(self.p, 0.2)
        self.sim = Simulator(world, start, adas_on=self.mode, params=self.p, aeb_config=aeb_cfg,
                             dynamics=Dynamics(), seed=seed or 0)
        self.driver = HumanLikeDriver(seed=seed or 7) if self.driver_kind == "virtual" else None
        self.estimator = IntentEstimator(self.intent_model)
        self.risk = {"base": 0.0, "blend": 0.0, "adaptive": 0.0}
        self._next_risk = 0.0
        self.accum = 0.0
        self.events = []
        self.crash_wall_time = None
        self._last_level = 0
        self.last_probs = self.estimator.probs
        for pid, val in self.tune_overrides.items():
            self._apply(pid, val)
        self._sent_scan = -1
        self.event("info", f"Loaded: {spec['name']}")

    def reset(self):
        self.load(self.scenario_id)

    def event(self, kind, text):
        self.events.append({"t": round(self.sim.t, 2), "kind": kind, "text": text})
        del self.events[:-40]

    # ------------------------------------------------------------------ tuning
    def _apply(self, pid, val):
        sim, cfg = self.sim, self.sim.adas.aeb.cfg
        group, name = pid.split(".")
        if group == "aeb":
            setattr(cfg, name, float(val))
        elif group == "model":
            model = dataclasses.replace(sim.adas.model, **{name: float(val)})
            sim.adas.model = sim.adas.aeb.model = sim.adas.estimator.model = model
        elif group == "truth":
            if name == "v_scale":
                base = SpeedModel()
                sim.dyn.speed_model = dataclasses.replace(sim.dyn.speed_model, v_max=base.v_max * float(val))
            else:
                setattr(sim.dyn, name, float(val))
        elif group == "sim":
            setattr(sim, name, float(val))
        elif group == "lidar":
            setattr(sim.lidar, name, float(val))
        elif group == "driver" and self.driver is not None:
            if name == "lapse_rate":
                self.driver.lapse_rate = float(val)
            elif name == "reaction":
                self.driver.reaction = float(val)

    def set_param(self, pid, val):
        if pid in TUNABLE_IDS:
            self.tune_overrides[pid] = float(val)
            self._apply(pid, val)

    def tunable_spec(self):
        cfg = self.sim.adas.aeb.cfg
        out = []
        for pid, label, group, lo, hi, step, unit, help_ in TUNABLES:
            g, n = pid.split(".")
            if pid in self.tune_overrides:
                cur = self.tune_overrides[pid]
            elif g == "aeb":
                cur = getattr(cfg, n)
            elif g == "model":
                cur = getattr(self.sim.adas.model, n)
            elif g == "truth":
                cur = 1.0 if n == "v_scale" else getattr(self.sim.dyn, n)
            elif g == "sim":
                cur = getattr(self.sim, n)
            elif g == "lidar":
                cur = getattr(self.sim.lidar, n)
            else:
                cur = self.driver.lapse_rate if (self.driver and n == "lapse_rate") else \
                    (self.driver.reaction if (self.driver and n == "reaction") else 0.12 if n == "lapse_rate" else 0.25)
            out.append({"id": pid, "label": label, "group": group, "min": lo, "max": hi, "step": step,
                        "unit": unit, "help": help_, "value": float(cur)})
        return out

    def export_tuning(self):
        cfg = self.sim.adas.aeb.cfg
        m = self.sim.adas.model
        return {"aeb": dataclasses.asdict(cfg), "speed_model": dataclasses.asdict(m),
                "vehicle": dataclasses.asdict(self.p), "lidar_min_range": self.sim.lidar.min_range}

    # ------------------------------------------------------------------ assists
    def teleport(self, x, y, theta_deg=0.0):
        """Place the car (for tests and demos)."""
        c = self.sim.car
        c.x, c.y, c.theta, c.v = float(x), float(y), math.radians(float(theta_deg)), 0.0
        c.collided = False
        self.sim.queue.clear()
        self.crash_wall_time = None
        self.sim.adas.parking.stop()

    def toggle_park(self):
        park = self.sim.adas.parking
        if park.active:
            park.stop()
            self.event("info", "Auto-park cancelled")
        else:
            park.start()
            self.event("info", "Auto-park started")

    def cycle_lane(self):
        lane = self.sim.adas.lane
        lane.mode = {"off": "warn", "warn": "assist", "assist": "off"}[lane.mode]
        self.event("info", f"Lane keeping: {lane.mode}")

    def toggle_acc(self):
        acc = self.sim.adas.acc
        acc.enabled = not acc.enabled
        self.event("info", f"Follow-the-leader {'on' if acc.enabled else 'off'}")

    def toggle_isa(self):
        isa = self.sim.adas.isa
        isa.enabled = not isa.enabled
        self.event("info", f"Sign adaptation {'on' if isa.enabled else 'off'}")

    def cv_frame(self, jpeg_bytes):
        """A frame rendered by the browser's virtual camera: run real OpenCV ArUco detection on it."""
        import cv2
        from adas.markers import detect_image
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        sizes = {o.marker_id: o.size for o in self.world.objects if hasattr(o, "marker_id")}
        found = [o for o in detect_image(frame, self.sim.marker_sensor.cam, sizes) if o.id in sizes]   # ignore unknown IDs
        self.cv_dets = [{"id": o.id, "dist": round(o.dist, 3), "x": round(o.x, 3), "y": round(o.y, 3),
                         "corners": o.corners.round(1).tolist()} for o in found]
        self.cv_time = time.time()

    # ------------------------------------------------------------------ control
    def set_mode(self, mode):
        if mode in ("off", "advisory", "active"):
            self.mode = mode
            self.sim.adas_mode = mode
            self.event("info", f"ADAS mode: {mode}")

    def set_input(self, steer, pwm):
        self.input = (max(-1.0, min(1.0, float(steer))), max(-255.0, min(255.0, float(pwm))))

    def toggle_record(self):
        if self.recording is None:
            self.recording = IntentLogger(rate_hz=20.0)
            self.sim.logger = self.recording
            self.event("info", "Recording driving data")
        else:
            rows = len(self.recording.rows)
            path = os.path.join(MODEL_DIR, f"drive_{int(time.time())}.csv")
            os.makedirs(MODEL_DIR, exist_ok=True)
            self.recording.path = path
            self._write_rows(path, self.recording.rows)
            self.sim.logger = None
            self.recording = None
            self.event("info", f"Saved {rows} rows to {os.path.basename(path)}")

    @staticmethod
    def _write_rows(path, rows):
        import csv
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _command(self):
        if self.driver is not None:
            return self.driver(self.sim.t, self.sim)
        return self.input

    def tick(self, real_dt):
        if self.paused:
            return
        sim = self.sim
        self.accum += min(real_dt, 0.1) * self.speed
        n = min(int(self.accum / DT), 80)
        self.accum -= n * DT

        for _ in range(n):
            if sim.car.collided:
                if self.crash_wall_time is None:
                    self.crash_wall_time = time.time()
                    self.event("crash", f"CRASH at {sim.car.impact_speed:.2f} m/s")
                elif time.time() - self.crash_wall_time > 3.0:
                    self.reset()
                    return
                break
            steer, pwm = self._command()
            sim.step(steer, pwm)
            probs = self.estimator.update(sim.t, steer, pwm, sim.adas)
            self.last_probs = probs
            if sim.t >= self._next_risk and self.estimator.row is not None:
                self._next_risk = sim.t + 0.1
                row, rb, rblend = warning_row(self.estimator.row, sim.adas, steer, pwm, probs, self.scorer)
                self.risk = {"base": rb, "blend": rblend,
                             "adaptive": self.adaptive.risk(row, rb) if self.adaptive else 0.0}
            level = sim.level
            if level != self._last_level:
                if level == 3:
                    self.event("brake", "Auto-braking engaged")
                elif level == 2 and self._last_level < 2:
                    self.event("warn", "Collision warning")
                self._last_level = level

    # ------------------------------------------------------------------ output
    def describe(self):
        return {"type": "world", "scenario": self.scenario_id, "name": self.spec["name"],
                "text": self.spec["text"], "driver": self.driver_kind,
                "objects": self.world.describe_static(),
                "vehicle": {"wheelbase": self.p.wheelbase, "track": self.p.pivot_track,
                            "width": self.p.width, "front": self.p.front_x, "rear": self.p.rear_x,
                            "lidar_x": self.p.lidar_x}}

    def _park_error(self):
        """Final pose error against the bay's true centre line, once parked (simulator ground truth)."""
        a = self.sim.adas
        if a.parking.state != "done":
            return None
        marker = next((o for o in self.world.objects if hasattr(o, "marker_id") and o.marker_id == a.parking.marker_id), None)
        if marker is None:
            return None
        c = self.sim.car
        lat, yaw = a.parking.error(c.x, c.y, c.theta, marker)
        return {"lateral_cm": round(lat * 100, 1), "yaw_deg": round(math.degrees(yaw), 1)}

    def features(self):
        a = self.sim.adas
        lead = a.acc.lead
        return {"acc": {"on": a.acc.enabled, "lead": [round(lead[0], 2), round(lead[1], 2)] if lead else None},
                "isa": {"on": a.isa.enabled, "active": a.isa.active,
                        "cap": round(a.isa.cap, 2) if math.isfinite(a.isa.cap) else -1},
                "park": {"state": a.parking.state, "msg": a.parking.message, "error": self._park_error()},
                "fault": a.fault,
                "lane": {"mode": a.lane.mode, "valid": a.lane.est.valid, "offset": round(a.lane.est.offset, 3),
                         "heading": round(a.lane.est.heading, 3), "level": a.lane.level, "assisting": a.lane.assisting}}

    def snapshot(self, last_scan=-1):
        sim, c, p = self.sim, self.sim.car, self.p
        wl, wr = steer_to_wheel_angles(sim.applied_steer, p)
        info = sim.info

        def fin(x, default=-1.0):
            return round(x, 3) if isinstance(x, (int, float)) and math.isfinite(x) else default

        # predicted path in world coordinates
        delta = steer_to_delta(sim.applied_steer, p)
        direction = info.get("direction", 1) if abs(sim.estimator.v) > 0.03 else (
            -1 if sim.driver_pwm < 0 else 1)
        D = info.get("D", math.inf)
        length = min(D, 1.6) if math.isfinite(D) else 1.6
        s = direction * np.linspace(0.0, max(length, 0.05), 32)
        gx, gy, _ = path_pose(delta, s, p)
        ct, st = math.cos(c.theta), math.sin(c.theta)
        path = np.column_stack([c.x + ct * gx - st * gy, c.y + st * gx + ct * gy]).round(3).tolist()

        snap = {
            "type": "state", "t": round(sim.t, 2),
            "car": {"x": round(c.x, 4), "y": round(c.y, 4), "th": round(c.theta, 4), "v": round(c.v, 3),
                    "steer": round(sim.applied_steer, 3), "wl": round(wl, 4), "wr": round(wr, 4),
                    "collided": bool(c.collided), "dist": round(sim.distance, 2)},
            "cmd": {"steer": round(sim.driver_steer, 3), "pwm": round(sim.driver_pwm, 1),
                    "out": round(sim.pwm_out, 1)},
            "adas": {"mode": self.mode, "level": sim.level, "D": fin(info.get("D", math.inf)),
                     "D_static": fin(info.get("D_static", math.inf)), "D_moving": fin(info.get("D_moving", math.inf)),
                     "D_aware": fin(info.get("D_aware", math.inf)), "v_safe": fin(info.get("v_safe", math.inf)),
                     "v_aware": fin(info.get("v_aware", math.inf)), "ttc": fin(info.get("ttc", math.inf)),
                     "v_est": round(sim.estimator.v, 3), "v_lidar": fin(info.get("v_lidar", 0.0), 0.0)},
            "path": path,
            "dyn": self.world.describe_dynamic(),
            "intent": {k: round(v, 3) for k, v in self.last_probs.items()},
            "risk": {k: round(v, 3) for k, v in self.risk.items()},
            "metrics": {"min_clearance": fin(sim.min_clearance * 100.0, -1.0), "brake_time": round(sim.brake_time, 2),
                        "distance": round(sim.distance, 2)},
            "events": self.events[-8:],
            "features": self.features(),
            "markers": [{"id": o.id, "dist": round(o.dist, 3), "x": round(o.x, 3), "y": round(o.y, 3),
                         "corners": o.corners.round(1).tolist()} for o in sim.marker_sensor.observations],
            "cv": self.cv_dets if time.time() - self.cv_time < 1.5 else [],
            "speed": self.speed, "paused": self.paused, "recording": self.recording is not None,
            "scan_id": sim.scan_id,
        }

        # tracked objects, converted to world coordinates
        sx, sy, sth = sim.scan_pose
        cs, ss = math.cos(sth), math.sin(sth)
        tracks = []
        for tr in sim.adas.tracks:
            x, y = tr.pos
            vx, vy = tr.vel_obj
            tracks.append({"id": tr.id, "x": round(sx + cs * x - ss * y, 3), "y": round(sy + ss * x + cs * y, 3),
                           "vx": round(cs * vx - ss * vy, 3), "vy": round(ss * vx + cs * vy, 3),
                           "r": round(tr.radius, 3), "moving": bool(tr.moving)})
        snap["tracks"] = tracks

        if sim.scan_id != last_scan and len(sim.points):
            pts = sim.points[::2]
            wx = sx + cs * pts[:, 0] - ss * pts[:, 1]
            wy = sy + ss * pts[:, 0] + cs * pts[:, 1]
            snap["lidar"] = np.column_stack([wx, wy]).round(3).ravel().tolist()
        return snap
