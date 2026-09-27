"""Control panel for the RC car (runs on the Pi as the rc-panel service, port 8080, as root).
One button per job; only one job owns the LiDAR/ESP32 at a time:
  drive  - the WiFi safety relay (rc-relay service) using the safety thresholds set here
  oa     - obstacle-avoidance demo (pi/bypass_run.py) with the tuning set here
  lidar / center / speed / turn - the calibration scripts
The live LiDAR view is whatever serves port 8090 (relay GUI or the running script's GUI)."""
import http.server
import json
import os
import shutil
import signal
import subprocess
import threading
import time

ROOT = "/home/pi/rc_car"
PI = ROOT + "/pi"
PORT = 8080
TUNING = PI + "/tuning_real_car.json"
PARAMS = PI + "/panel_params.json"
RESULTS = PI + "/cal_results.json"
CHILD_LOG = "/home/pi/panel_child.log"

MODES = {
    "drive": {"title": "Drive with safety stops", "script": None,
              "prep": "Starts the safety relay. Drive with rc_controller.py on the laptop. Change thresholds below, then press again to apply."},
    "oa": {"title": "Obstacle avoidance demo", "script": "pi/bypass_run.py",
           "prep": "Put ONE obstacle straight ahead (about 1.4 m from the car's start spot), with about 3.5 m clear along the line and room on both sides. The car backs up to get run-up first."},
    "all": {"title": "Run everything", "script": "pi/run_all.py",
            "prep": "Runs LiDAR front, steering centre, speed + stopping, turning and the obstacle-avoidance demo in order. It pauses and tells you how to set up the arena before each group; nothing is applied automatically."},
    "logdrive": {"title": "Logging drive (simulator data)", "script": "pi/log_drive.py",
                 "prep": "The car drives short legs forward and back at several speeds and steering angles, checking the LiDAR for room before each leg. Needs about 1.5 m clear ahead and behind. About 2 minutes. Everything is logged for fitting the simulator."},
    "lidar": {"title": "Calibrate LiDAR front", "script": "pi/lidar_front_cal.py",
              "prep": "Put ONE object dead-centre in front of the car, 0.3-1.2 m away. The car does not move."},
    "center": {"title": "Calibrate steering centre", "script": "pi/center_fine.py",
               "prep": "The car moves: needs an open area (about 1.5 m clear ahead, 0.5 m behind). Takes about 3-4 minutes."},
    "speed": {"title": "Calibrate speed + stopping", "script": "pi/brake_run.py",
              "prep": "The car moves: needs about 1.6 m clear ahead of its start (it backs up first). Five speed levels, about 2 minutes."},
    "turn": {"title": "Calibrate turning", "script": "pi/turn_test.py",
             "prep": "The car moves in arcs: needs an open area, 1.5 m in front and free space to each side. About 3 minutes."},
}

# name, label, min, max, step, unit, help
OA_FIELDS = [
    ("pwm", "Cruise speed (PWM)", 90, 180, 5, "", "throttle while avoiding; higher = less time to steer"),
    ("min_detect_dist_m", "Refuse closer than", 0.4, 1.5, 0.05, "m", "obstacle nearer than this at first sight: stop instead of a sidestep that can't finish"),
    ("look_m", "Detect obstacle from", 0.6, 2.0, 0.05, "m", "how far ahead an obstacle in the path is noticed"),
    ("half_path", "Path half-width", 0.08, 0.35, 0.01, "m", "obstacle counts as 'in the way' within this of the line"),
    ("clear_m", "Side clearance", 0.10, 0.45, 0.01, "m", "distance kept between the obstacle edge and the car centreline"),
    ("min_gap_m", "Minimum free gap", 0.25, 0.9, 0.05, "m", "a side is only used if the free gap beside the obstacle is at least this"),
    ("l_h", "Steer-back length", 0.25, 1.2, 0.05, "m", "smaller = sharper steering back to the line"),
    ("tau_s", "Steering response", 0.10, 0.8, 0.05, "m", "smaller = stronger/faster steering, more overshoot risk"),
    ("max_servo_deg", "Max steering angle", 8, 30, 1, "deg", "limit on servo deflection from centre"),
    ("extra_straight_m", "Straight after rejoin", 0.0, 1.0, 0.05, "m", "keeps driving straight this far after regaining the line"),
    ("hw_front_stop_m", "Hard front stop", 0.15, 0.6, 0.01, "m", "independent LiDAR emergency stop: front closer than this"),
    ("servo_slew", "Steering slew limit", 20, 200, 5, "deg/s", "how fast the steering may move (no snapping)"),
]
OA_DEFAULTS = {"pwm": 115, "min_detect_dist_m": 0.75, "look_m": 1.15, "half_path": 0.17, "clear_m": 0.20, "min_gap_m": 0.45,
               "l_h": 0.45, "tau_s": 0.20, "max_servo_deg": 24, "extra_straight_m": 0.35, "hw_front_stop_m": 0.28, "servo_slew": 60}

SAFETY_FIELDS = [
    ("margin", "Stop margin", 0.0, 0.4, 0.01, "m", "extra gap kept in front of an obstacle after braking"),
    ("footprint_margin", "Body clearance margin", 0.0, 0.2, 0.01, "m", "widens the car footprint used for obstacle checks"),
    ("decel", "Assumed braking decel", 0.4, 3.0, 0.1, "m/s2", "lower = brakes earlier (more conservative)"),
    ("latency", "Reaction latency", 0.05, 0.6, 0.01, "s", "command + motor delay assumed in the stopping distance"),
    ("brake_pwm_max", "Active brake strength", 0, 255, 5, "PWM", "peak reverse-brake PWM"),
    ("max_pwm", "Max throttle", 60, 255, 5, "PWM", "cap on forward throttle the relay lets through"),
    ("aware_margin", "Awareness margin", 0.0, 0.6, 0.02, "m", "extra caution zone around the footprint"),
    ("scan_timeout", "LiDAR timeout", 0.2, 1.5, 0.05, "s", "no fresh scan for this long = stop"),
]

lock = threading.Lock()
state = {"mode": None, "proc": None, "since": None, "message": "idle"}


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return str(e)


def read_json(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def write_json(path, data):
    tmp = path + ".tmp"
    json.dump(data, open(tmp, "w"), indent=2)
    os.replace(tmp, path)
    try:
        shutil.chown(path, "pi", "pi")
    except Exception:
        pass


def relay_active():
    return sh(["systemctl", "is-active", "rc-relay"]) == "active"


def stop_all():
    """Stop the running script (SIGTERM -> its cleanup stops motors), the relay, then zero the motor."""
    p = state["proc"]
    if p and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            for _ in range(60):
                if p.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    state["proc"] = None
    if relay_active():
        sh(["systemctl", "stop", "rc-relay"])
    time.sleep(0.8)
    sh(["runuser", "-u", "pi", "--", "python3", PI + "/estop.py"], timeout=15)   # motor to zero, whoever held it
    state["mode"] = None


def apply_safety(vals):
    tun = read_json(TUNING, {})
    bak = TUNING + ".bak_panel"
    if not os.path.exists(bak):
        shutil.copy(TUNING, bak)
    names = {f[0] for f in SAFETY_FIELDS}
    for k, v in vals.items():
        if k in names:
            tun.setdefault("aeb", {})[k] = float(v)
    write_json(TUNING, tun)


def start(mode, params):
    if mode not in MODES:
        return "unknown mode"
    with lock:
        stop_all()
        pp = read_json(PARAMS, {})
        if mode == "oa":
            new = {k: float(v) for k, v in (params or {}).items() if k in OA_DEFAULTS}
            if new:
                pp.setdefault("oa", {}).update(new)
                write_json(PARAMS, pp)
        if mode == "drive":
            if params:
                pp["safety"] = {k: float(v) for k, v in params.items()}
                write_json(PARAMS, pp)
                apply_safety(pp["safety"])
            sh(["systemctl", "restart", "rc-relay"])
            state.update(mode="drive", since=time.time(), message="safety relay running - drive from the laptop controller")
            return "ok"
        for f in ("run_all_state.json", "run_all_continue.flag"):
            try:
                os.remove(os.path.join(PI, f))
            except OSError:
                pass
        log = open(CHILD_LOG, "w")
        state["proc"] = subprocess.Popen(["runuser", "-u", "pi", "--", "python3", "-u", ROOT + "/" + MODES[mode]["script"]],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(mode=mode, since=time.time(), message=MODES[mode]["title"] + " running")
        return "ok"


def apply_result(kind):
    res = read_json(RESULTS, {})
    tun = read_json(TUNING, {})
    pp = read_json(PARAMS, {})
    bak = TUNING + ".bak_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy(TUNING, bak)
    if kind == "lidar" and "lidar" in res:
        tun["mount"]["yaw_offset_deg"] = round(res["lidar"]["yaw_offset_deg"], 2)
        msg = f"LiDAR yaw offset set to {tun['mount']['yaw_offset_deg']}"
    elif kind == "servo_center" and "servo_center" in res:
        c = round(res["servo_center"]["servo_center"], 1)
        tun["servo"]["left_center"] = tun["servo"]["right_center"] = c
        pp.setdefault("oa", {})["servo_straight"] = c
        msg = f"servo centre set to {c}"
    elif kind == "speed" and "speed" in res:
        r = res["speed"]
        tun["speed_model"]["v_max"] = round(r["v_max"], 3)
        tun["speed_model"]["deadband"] = round(r["deadband"], 1)
        msg = f"speed model set: v_max {r['v_max']:.2f} m/s, deadband {r['deadband']:.0f}"
    elif kind == "turn" and "turn" in res:
        pp.setdefault("oa", {})["k_curv_per_deg"] = round(res["turn"]["k_curv_per_deg"], 4)
        msg = f"turn gain set to {res['turn']['k_curv_per_deg']:.4f} rad/m per servo degree (used by obstacle avoidance)"
    else:
        return "no such result yet"
    write_json(TUNING, tun)
    write_json(PARAMS, pp)
    return msg + "  (press Drive again to use it in the relay)"


def snapshot():
    p = state["proc"]
    running = state["mode"] == "drive" and relay_active()
    if p is not None:
        running = p.poll() is None
        if not running and state["mode"]:
            state["message"] = MODES[state["mode"]]["title"] + " - finished"
    pp = read_json(PARAMS, {})
    tun = read_json(TUNING, {})
    saf_def = {f[0]: tun.get("aeb", {}).get(f[0]) for f in SAFETY_FIELDS}
    try:
        tail = open(CHILD_LOG).read().splitlines()[-12:]
    except Exception:
        tail = []
    return {"mode": state["mode"], "running": running, "message": state["message"], "relay": relay_active(),
            "modes": {k: {"title": v["title"], "prep": v["prep"]} for k, v in MODES.items()},
            "oa_fields": OA_FIELDS, "safety_fields": SAFETY_FIELDS,
            "oa": {**OA_DEFAULTS, **pp.get("oa", {})}, "oa_defaults": OA_DEFAULTS,
            "safety": {**saf_def, **pp.get("safety", {})}, "safety_defaults": saf_def,
            "results": read_json(RESULTS, {}), "log": tail,
            "run_all": read_json(PI + "/run_all_state.json", None) if state["mode"] == "all" else None}


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(open(PI + "/control_panel.html", "rb").read(), "text/html")
        elif self.path.startswith("/api/state"):
            self._send(json.dumps(snapshot()).encode())
        else:
            self._send(b"{}", code=404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if self.path == "/api/start":
            out = start(body.get("mode"), body.get("params"))
        elif self.path == "/api/stop":
            with lock:
                stop_all()
                state["message"] = "stopped - motors zeroed"
            out = "stopped"
        elif self.path == "/api/continue":
            open(PI + "/run_all_continue.flag", "w").close()
            out = "continuing"
        elif self.path == "/api/apply":
            out = apply_result(body.get("kind"))
        else:
            out = "unknown"
        self._send(json.dumps({"result": out}).encode())


if __name__ == "__main__":
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print(f"control panel on :{PORT}", flush=True)
    srv.serve_forever()
