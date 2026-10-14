"""Careful real-car checks through the real dashboard (gui/ev.py EVWindow) against the car's relay.

    python tools/car_ui_test.py --host 192.168.1.15 [scenarios...]     (default: idle steer brake reverse)

Sends exactly what rc_controller.py sends (its own servo maths, 50 Hz) to the relay - so the relay's gate and assists act as
in a real drive - and saves dashboard screenshots to reports/car_ui_test/. Extra safety layer on top of the relay, for testing
in a confined space: throttle capped at MAX_PWM, every motion time-limited, motor zero at once if the car is faster than
MAX_V or a LiDAR point comes within NEAR_M of the front bumper. The user's zones / lane files are never written. Never run
it while rc_controller.py is running.
"""
import argparse
import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "car_ui_test")
MAX_PWM = 50           # throttle cap (of 255): a crawl (~0.15 m/s; 70 reached 0.25 m/s)
MAX_V = 0.30           # m/s: faster than this -> motor zero
NEAR_M = 0.08          # m: a point this close to the front bumper while going forward -> motor zero


class Driver:
    """rc_controller.py's packets (steer -1..1, + = stick right; throttle PWM, + = forward) at 50 Hz to the relay."""

    def __init__(self, host):
        sys.argv = sys.argv[:1]
        import rc_controller as rc
        self.rc, self.host = rc, host
        self.steer, self.throttle, self.stop, self.cut = 0.0, 0, False, None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            thr = 0 if self.cut else int(max(-MAX_PWM, min(MAX_PWM, self.throttle)))
            l, r = self.rc.steering_to_servo_angles(self.rc.STEERING_SIGN * self.steer)
            self.sock.sendto(self.rc.build_command(l, r, thr).encode(), (self.host, 4210))
            time.sleep(0.02)


class CarTest:
    def __init__(self, win, driver, name):
        self.win, self.d, self.name = win, driver, name
        self.result = {"scenario": name}
        self.shots, self.trace = [], []
        self.mp = win.panels["map"][1]

    def state(self):
        return self.win.link.latest or {}

    def shot(self, tag):
        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, f"{self.name}_{len(self.shots):02d}_{tag}.png")
        self.win.grab().save(p)
        self.shots.append(p)

    def front_clear_m(self):
        """Smallest distance of the current scan's points ahead of the bumper, within the car's width + 3 cm."""
        from gui.dashboard import GEO, polar_xy
        v = polar_xy(self.state().get("_pts"))
        if not len(v):
            return None
        m = (v[:, 0] > GEO["front"] - 0.02) & (np.abs(v[:, 1]) < GEO["width"] / 2 + 0.03)
        return float((v[m, 0] - GEO["front"]).min()) if m.any() else None

    def fresh(self):
        """The dashboard has live data from the relay (newer than 0.3 s)."""
        return bool(self.win.link.latest) and time.time() - self.win.link.latest_t < 0.3

    def watch(self):
        """The test's own safety layer; returns the reason if it cut the motor."""
        st = self.state()
        if not self.fresh():
            self.d.cut = "no fresh data from the relay"
            return self.d.cut
        v = float((st.get("drive") or {}).get("v") or 0.0)
        near = self.front_clear_m()
        gate = st.get("gate") or {}
        pose = (st.get("world") or {}).get("pose")
        self.trace.append({"t": round(time.time() - self.t0, 2), "throttle": self.d.throttle, "v": round(v, 3), "pose": pose,
                           "front_clear_m": None if near is None else round(near, 3), "gate": gate.get("action"),
                           "free_m": gate.get("free_m"), "health": (st.get("health") or {}).get("state")})
        if abs(v) > MAX_V:
            self.d.cut = f"speed {v:.2f} m/s"
        elif self.d.throttle > 0 and near is not None and near < NEAR_M:
            self.d.cut = f"object {near * 100:.0f} cm from the bumper"
        elif (st.get("health") or {}).get("state") == "fault":
            self.d.cut = "relay health FAULT"
        return self.d.cut


def idle(t):
    """Standing still: stream, health, ESP32, the object straight ahead on the map."""
    t.d.throttle, t.d.steer = 0, 0.0
    yield 3.0
    st = t.state()
    t.result.update(health=st.get("health"), esp32=(st.get("esp32") or {}).get("state"), front_clear_m=t.front_clear_m())
    t.shot("idle")


def steer(t):
    """Full lock left, full lock right, centre - the car does not move (throttle 0)."""
    for name, s in (("left", -1.0), ("right", 1.0), ("centre", 0.0)):
        t.d.steer = s
        yield 1.5
        st = t.state()
        t.result[f"servo_{name}"] = [st.get("steer_a1"), st.get("steer_a2")]
        t.shot(f"steer_{name}")


def brake(t):
    """Crawl straight at the object ahead: the relay's gate must stop the car before it, without the test's own cut."""
    t.d.steer = 0.0
    while not t.fresh():
        yield 0.2
    yield 1.0
    t.result["start_clear_m"] = t.front_clear_m()
    t.result["start_pose"] = (t.state().get("world") or {}).get("pose")
    t.d.throttle = MAX_PWM
    t_end, stopped = time.time() + 15.0, 0
    while time.time() < t_end:
        if t.watch():
            break
        g = (t.state().get("gate") or {}).get("action") or ""
        stopped = stopped + 1 if ("holding" in g or "stopped" in g) else 0
        if stopped >= 20:                                 # the gate has held the car for ~1 s: done
            break
        yield 0.05
    t.shot("brake_end")
    t.d.throttle = 0
    yield 1.0
    t.result["end_clear_m"] = t.front_clear_m()
    t.result["end_pose"] = (t.state().get("world") or {}).get("pose")
    t.result["gate_actions"] = sorted({r["gate"] for r in t.trace if r["gate"]})
    t.result["max_v"] = max((abs(r["v"]) for r in t.trace), default=0.0)
    t.result["test_cut"] = t.d.cut


def reverse(t):
    """A short crawl backwards (the rear was ~1 m free), then stop."""
    t.d.steer = 0.0
    while not t.fresh():
        yield 0.2
    t.d.throttle = -MAX_PWM
    t_end = time.time() + 1.0
    while time.time() < t_end:
        if t.watch():
            break
        yield 0.05
    t.d.throttle = 0
    yield 1.0
    t.result["end_clear_m"] = t.front_clear_m()
    t.result["max_v"] = max((abs(r["v"]) for r in t.trace), default=0.0)
    t.result["test_cut"] = t.d.cut
    t.shot("reverse_end")


SCENARIOS = {"idle": idle, "steer": steer, "brake": brake, "reverse": reverse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("names", nargs="*", default=["idle", "steer", "brake", "reverse"])
    a = ap.parse_args()
    import gui.ev as EV
    os.makedirs(OUT, exist_ok=True)
    EV.ZONES_FILE = os.path.join(OUT, "zones_test.json")      # the user's gui/zones.json is never written
    EV.LANE_FILE = os.path.join(OUT, "lane_test.json")         # ... nor gui/parking_lane.json
    from PySide6 import QtCore, QtWidgets
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from gui import theme
    from gui.controls import gui_thread_gc
    from gui.dashboard import Link
    gui_thread_gc()
    theme.load_mode()
    app.setStyleSheet(theme.STYLE)
    driver = Driver(a.host)
    win = EV.EVWindow(Link(a.host))
    win.resize(1700, 1000)
    win.show()
    win.toggle("map", True)
    results, queue = [], list(a.names)
    box = {"gen": None, "test": None, "wake": time.time() + 4.0}

    def tick():
        if box["test"] is not None and box["gen"] is not None:
            box["test"].watch() if driver.throttle else None
        if time.time() < box["wake"]:
            return
        if box["gen"] is None:
            if not queue:
                driver.throttle, driver.steer = 0, 0.0
                timer.stop()
                QtCore.QTimer.singleShot(600, app.quit)
                return
            t = CarTest(win, driver, queue.pop(0))
            t.t0 = time.time()
            driver.cut = None
            box["test"], box["gen"] = t, SCENARIOS[t.name](t)
        try:
            box["wake"] = time.time() + float(next(box["gen"]))
        except StopIteration:
            driver.throttle = 0
            r = box["test"].result
            r["shots"] = [os.path.relpath(p, ROOT) for p in box["test"].shots]
            r["trace"] = box["test"].trace
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("trace",)}, default=str), flush=True)
            box["gen"], box["wake"] = None, time.time() + 1.5
        except Exception as e:
            driver.throttle = 0
            import traceback
            traceback.print_exc()
            results.append({"scenario": box["test"].name, "error": repr(e)})
            box["gen"] = None

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(20)
    try:
        app.exec()
    finally:
        driver.throttle = 0
        time.sleep(0.2)
        driver.stop = True
        json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
