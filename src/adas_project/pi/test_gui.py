"""Browser GUI for the standalone calibration/test scripts (the relay is stopped while they
run, so its own GUI isn't available). Serves the same lidar_gui.html on port 8090 plus a
live test-status panel: what is being tested, progress, and a rolling result log."""
import atexit
import http.server
import json
import os
import signal
import threading
import time

PORT = 8090
RESULTS = "/home/pi/rc_car/pi/cal_results.json"


def save_result(key, data):
    """Merge one calibration result into cal_results.json (the control panel offers 'Apply')."""
    try:
        cur = json.load(open(RESULTS))
    except Exception:
        cur = {}
    data = dict(data)
    data["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cur[key] = data
    json.dump(cur, open(RESULTS, "w"), indent=2)


class TestGui:
    def __init__(self, rig, port=PORT):
        self.rig = rig
        # the control panel stops a test with SIGTERM: unwind normally so motors stop and the LiDAR closes
        try:
            signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
        except ValueError:
            pass

        def _cleanup():
            try:
                rig.stop()
                rig.close()
            except Exception:
                pass
        atexit.register(_cleanup)
        self.state = {"title": "idle", "message": "", "progress": None, "log": [],
                      "activity": "", "servo": None, "alert": ""}
        self.lock = threading.Lock()
        gui = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lidar_gui.html")
                    self._send(open(p, "rb").read(), "text/html")
                elif self.path.startswith("/api/scan"):
                    self._send(json.dumps(gui.snapshot()).encode(), "application/json")
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                self.send_response(200)
                self.end_headers()

        for _ in range(40):          # an earlier test may still be showing its results page
            try:
                self.server = http.server.ThreadingHTTPServer(("0.0.0.0", port), H)
                break
            except OSError:
                time.sleep(1.0)
        else:
            self.server = None
            print("test GUI: port busy, continuing without it", flush=True)
            return
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"test GUI on http://<pi-ip>:{port}/", flush=True)

    def set(self, title=None, message=None, progress=None, activity=None, servo=None):
        with self.lock:
            if title is not None:
                self.state["title"] = title
            if message is not None:
                self.state["message"] = message
            if progress is not None:
                self.state["progress"] = progress
            if activity is not None:
                self.state["activity"] = activity
            if servo is not None:
                self.state["servo"] = servo

    def log(self, line):
        with self.lock:
            self.state["log"] = (self.state["log"] + [f"{time.strftime('%H:%M:%S')}  {line}"])[-14:]
        print(line, flush=True)

    def snapshot(self):
        r = self.rig
        front, rear = r.front(), r.rear()
        with self.lock:
            test = json.loads(json.dumps(self.state))
        if front is not None and front < 0.55:
            test["alert"] = f"OBSTACLE AHEAD  {front:.2f} m"
        elif rear is not None and rear < 0.30:
            test["alert"] = f"OBSTACLE BEHIND  {rear:.2f} m"
        else:
            test["alert"] = ""
        test["scan_ok"] = r.is_fresh()
        return {
            "t": time.time(), "mode": "test", "test": test,
            "front_dist": front, "rear_dist": rear,
            "front_blocked": front is not None and front < 0.35,
            "rear_blocked": rear is not None and rear < 0.35,
            "body_min_front": None, "body_min_rear": None,
            "body_alert_front": False, "body_alert_rear": False,
            "steer_a1": 90 + r._steer_offset, "steer_a2": 90 + r._steer_offset,
            "pwm_sent": int(r._pwm_now), "braking": False,
            "tracks": [], "points": r.points(),
        }
