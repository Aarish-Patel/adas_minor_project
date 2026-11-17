"""Demo clips of every feature, recorded from the REAL dashboard (gui/ev.py EVWindow) driving the twin relay (tools/sim_car.py), the way
tools/hmi_test.py drives it: map clicks, tiles, the dead-man throttle sent as rc_controller.py packets. Each scenario is written as an
MP4 of the dashboard window (or the lab window) at ~8 frames/s.

    python tools/demo_videos.py [clip ...]     -> reports/demo_videos/<clip>.mp4 (+ a poster .png each)

Clips: overview, goto_home, emergency_brake, pedestrian, evasive_steer, speed_zone, parallel_park, bay_park, monte_carlo, ml_training.
Uses its own twin relay on the test ports (UDP 4310 / HTTP 8190), never the car or a relay the user has open.
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hmi_test as H                                                      # noqa: E402

OUT = os.path.join(ROOT, "reports", "demo_videos")
FPS, WIDTH = 8, 1280


class Recorder:
    """Grabs a widget every 1/FPS s into an MP4 (OpenCV, mp4v)."""

    def __init__(self):
        self.writer, self.target, self.path, self.frames = None, None, None, 0

    def start(self, name, target):
        import cv2
        os.makedirs(OUT, exist_ok=True)
        self.target, self.path, self.frames = target, os.path.join(OUT, f"{name}.mp4"), 0
        img = self._grab()
        h = int(round(img.shape[0] * WIDTH / img.shape[1])) // 2 * 2
        self.size = (WIDTH, h)
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, self.size)

    def _grab(self):
        from PySide6.QtGui import QImage
        q = self.target.grab().toImage().convertToFormat(QImage.Format_RGB32)
        w, h = q.width(), q.height()
        a = np.frombuffer(q.constBits(), np.uint8, count=q.sizeInBytes()).reshape(h, q.bytesPerLine() // 4, 4)[:, :w, :3]
        return np.ascontiguousarray(a)

    def tick(self):
        if self.writer is None:
            return
        import cv2
        f = cv2.resize(self._grab(), self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(f)
        self.frames += 1
        if self.frames == FPS * 4:
            cv2.imwrite(self.path[:-4] + ".png", f)

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            print(f"  wrote {os.path.relpath(self.path, ROOT)} ({self.frames / FPS:.0f} s)", flush=True)
        self.writer = None


REC = Recorder()


def _post(test, path):
    try:
        test.win.link.http_post(path)
    except Exception as e:
        test.note(post_failed=path, error=repr(e))


def _origin(test):
    test.win.link.send("ORIGIN")


def _drive(test, throttle, secs, servo=87, step=0.2):
    test.driver.throttle, test.driver.servo = throttle, servo
    t_end = time.time() + secs
    while time.time() < t_end:
        yield step
    test.driver.throttle, test.driver.servo = 0, 87


def _room(test, room):
    yield from H._switch_room(test, room)


def _open_world(test, name):
    """A room that has no button on the dashboard: switched over HTTP, then the world frame set to the new start."""
    _post(test, f"/api/sim/world/{name}")
    yield 1.5
    _origin(test)
    yield 1.0


def clip(name, target="window"):
    """Decorator: record the scenario generator as <name>.mp4."""
    def wrap(fn):
        def run(test):
            test.win.toggle("map", True)
            yield 0.5
            REC.start(name, test.win if target == "window" else target(test))
            try:
                yield from fn(test)
            finally:
                yield 1.0
                REC.stop()
                _post(test, "/api/sim/clear_props")
                test.driver.throttle, test.driver.servo = 0, 87
                test.win.link.send("GOTO CANCEL")
        return run
    return wrap


@clip("overview")
def overview(test):
    yield from _room(test, "room")
    yield 2.0
    yield from _drive(test, 110, 4.0, servo=80)                      # a little manual driving: the map builds, the HMI follows
    yield from _drive(test, 110, 3.0, servo=95)
    yield 2.0


@clip("goto_home")
def goto_home(test):
    yield from H.goto_and_home(test)


@clip("emergency_brake")
def emergency_brake(test):
    yield from H.brake_wall(test)


@clip("pedestrian")
def pedestrian(test):
    yield from _open_world(test, "open")
    test.win.link.send("ASSIST moving ON")
    _post(test, "/api/sim/add/walker")                                # crosses 1.6 m ahead, from the right
    yield 0.3
    yield from _drive(test, 150, 9.0)                                 # the driver keeps the throttle on, straight at the crossing
    yield 2.0


@clip("evasive_steer")
def evasive_steer(test):
    yield from _open_world(test, "open")
    test.win.link.send("ASSIST evasive ON")                           # (off by default on the car: the driver switches it on)
    yield 0.5
    _post(test, "/api/sim/add/box")                                   # a box in the lane
    yield 0.5
    yield from _drive(test, 200, 10.0)                                # throttle held, no steering: the car steers round the box
    test.win.link.send("ASSIST evasive OFF")
    yield 2.0


@clip("speed_zone")
def speed_zone(test):
    yield from H.zone_limit(test)


@clip("parallel_park")
def parallel_park(test):
    yield from H.run_parking(test, 2.2)


@clip("bay_park")
def bay_park(test):
    """The car starts turned away from the bays (the marker out of the camera's view): the operator sets the general area in front of the
    bays and presses 'Find the marker and park' - the car drives there, finds the marker, lines up with the bay and parks nose in."""
    yield from _room(test, "park_perp")
    yield 1.0
    yield from _drive(test, -150, 2.6, servo=50)                     # back away on full lock: the camera now looks past the bays
    yield from _drive(test, 120, 1.2, servo=125)
    yield 1.5
    tiles = [w for w in test.win.findChildren(object) if hasattr(w, "_park_marker_or_area") and hasattr(w, "last_goto")]
    for w in tiles:
        w.last_goto = ((0.55, 0.0), None)                             # the area: in front of the middle bay (world frame = the start)
    yield 0.5
    test.shot("area_set")
    for w in tiles:
        w._park_marker_or_area()
    yield 0.5
    t_end = time.time() + 70.0
    test.driver.throttle = 150                                        # the dead-man throttle held through the whole manoeuvre
    while time.time() < t_end:
        st = test.state()
        nav = (st.get("nav") or {})
        ms = ((st.get("assist") or {}).get("marker_search")) or ""
        if nav.get("state") == "idle" and time.time() > t_end - 66 and nav.get("msg") in ("arrived", "parked") and not ms:
            break
        yield 0.3
    test.driver.throttle = 0
    yield 2.0


def bay_park_old(test):
    for w in test.win.findChildren(object):                           # (a Go-to area left by an earlier clip would turn the tile into PARK_AREA)
        if hasattr(w, "last_goto"):
            w.last_goto = None
    yield from H.bay_park(test)


def _lab(key):
    def target(test):
        test.win.open_lab(key)
        w = test.win.labs[key]
        w.resize(1600, 950)
        w.show()
        w.raise_()
        return w
    return target


@clip("monte_carlo", target=_lab("mc"))
def monte_carlo(test):
    yield 25.0


@clip("ml_training", target=_lab("train"))
def ml_training(test):
    yield 25.0


CLIPS = {"overview": overview, "goto_home": goto_home, "emergency_brake": emergency_brake, "pedestrian": pedestrian,
         "evasive_steer": evasive_steer, "speed_zone": speed_zone, "parallel_park": parallel_park, "bay_park": bay_park,
         "monte_carlo": monte_carlo, "ml_training": ml_training}


def main(argv):
    names = [a for a in argv if not a.startswith("--")] or list(CLIPS)
    H.SCENARIOS.clear()
    H.SCENARIOS.update({n: CLIPS[n] for n in names})
    H.OUT = os.path.join(OUT, "shots")
    from PySide6 import QtCore, QtWidgets
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rec_timer = QtCore.QTimer()
    rec_timer.timeout.connect(REC.tick)
    rec_timer.start(int(1000 / FPS))
    H.main(names + ["--keep-relay"] if "--keep-relay" in argv else names)


if __name__ == "__main__":
    main(sys.argv[1:])
