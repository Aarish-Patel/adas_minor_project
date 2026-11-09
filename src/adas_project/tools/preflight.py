"""Pre-flight check before showing the car (TODO AN5): reads the relay's live state (read-only: nothing is sent to the motor or the servos)
and prints what is ready, what to look at and what is wrong, with the fix.

    python tools/preflight.py --host 192.168.1.10          # the car (the Pi)       python tools/preflight.py --host 127.0.0.1   # the twin

Checks: relay answering, health, LiDAR rate, ESP32, Pi temperature, something sensible in front of / behind the car, the parking camera
service and whether it sees the bay marker, the control panel. Exit code 0 when nothing FAILS (warnings are for the presenter's eyes).
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

OK, WARN, FAIL = "ok  ", "WARN", "FAIL"


def get_json(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def evaluate(st, camera_service=None, panel_ok=True):
    """[(level, name, detail)] from a relay state packet `st` (dict, or None when nothing came), the camera service's own /state and whether the
    control panel answered. Pure: the unit tests feed it states."""
    out = []
    if not st:
        return [(FAIL, "relay", "no state from the relay: is the Pi on, the relay running (control panel -> Drive) and the IP right?")]
    out.append((OK, "relay", "answering"))
    h = st.get("health") or {}
    if h.get("state") == "normal":
        temp = h.get("temp_c")
        out.append((OK, "health", f"normal (LiDAR {h.get('lidar_hz') or 0:.0f} Hz" + (f", Pi {temp:.0f} C)" if temp is not None else ")")))
    else:
        causes = ", ".join(h.get("causes") or ["unknown"])
        fix = " - the LiDAR is not delivering scans: check its USB cable and its power (battery bank charged, motor spinning); the relay restarts " \
              "itself and re-finds it after 8 s" if "no LiDAR" in causes else ""
        out.append((FAIL if h.get("state") == "fault" else WARN, "health", f"{h.get('state', '?').upper()}: {causes}{fix}"))
    hz = h.get("lidar_hz")
    if hz is not None and h.get("state") == "normal" and hz < 8.0:
        out.append((WARN, "lidar rate", f"{hz:.1f} Hz (about 10-12 is normal): a low rate means a weak power supply or a loose cable"))
    e = st.get("esp32") or {}
    if e.get("state") == "ok":
        reboots = int(e.get("reboots") or 0)
        out.append((WARN if reboots > 1 else OK, "esp32", f"answering on {e.get('port', '?')}" + (f", {reboots} reboots since the relay started: "
                    f"check the motor / servo supply" if reboots > 1 else "")))
    else:
        out.append((FAIL, "esp32", f"{e.get('state', 'no answer')}: {e.get('detail', '')} - check its USB cable to the Pi"))
    t = h.get("temp_c")
    if t is not None and t >= 70:
        out.append((WARN if t < 80 else FAIL, "pi temperature", f"{t:.0f} C: the Pi throttles at 80 C - point a fan at it"))
    fd, rd = st.get("front_dist"), st.get("rear_dist")
    out.append((OK if fd and rd else WARN, "lidar view", f"front {fd if fd else '-'} m, rear {rd if rd else '-'} m" +
                ("" if fd and rd else " (nothing measured on one side: is the LiDAR window clear?)")))
    cam = st.get("camera") or {}
    if not cam:
        out.append((WARN, "camera", "this relay does not report a camera (older code): deploy the current relay to use bay parking"))
    elif not cam.get("service"):
        out.append((WARN, "camera", f"the {cam.get('facing', '?')} camera service is not running: start pi/marker_camera.py on the Pi (bay parking needs it)"))
    elif cam.get("seen"):
        out.append((OK, "camera", f"{cam.get('facing')} camera sees marker {cam.get('marker_id')} at {cam.get('dist')} m, {cam.get('bearing_deg', 0):+.0f} deg"))
    else:
        out.append((WARN, "camera", f"{cam.get('facing')} camera running, marker {cam.get('marker_id')} not in view: place the car facing the bay "
                    f"(marker at lens height, 0.3-1.8 m away, well lit)"))
    if camera_service is not None and camera_service.get("ok"):
        out.append((OK, "camera service", f"{camera_service.get('fps', 0):.0f} fps, {camera_service.get('pipeline_ms', 0):.0f} ms per frame"))
    elif camera_service is not None:
        out.append((WARN, "camera service", f"no frames: {camera_service.get('why', 'service answered without a picture')} - check the webcam's USB cable"))
    out.append((OK if panel_ok else WARN, "control panel", "answering" if panel_ok else "not answering on :8080 (the relay can still be driven)"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--wait", type=float, default=6.0, help="seconds to listen for the relay's state")
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    from gui.dashboard import Link
    link = Link(a.host)
    t0 = time.time()
    while time.time() - t0 < a.wait and not link.latest:
        time.sleep(0.2)
    time.sleep(1.5)
    st = dict(link.latest or {}) or None
    cam = get_json(f"http://{a.host}:8091/state")
    panel = get_json(f"http://{a.host}:8080/api/status") is not None
    rows = evaluate(st, cam, panel)
    for lvl, name, detail in rows:
        print(f"[{lvl}] {name:15s} {detail}")
    fails = sum(1 for lvl, _, _ in rows if lvl == FAIL)
    warns = sum(1 for lvl, _, _ in rows if lvl == WARN)
    print(f"\n{'READY' if not fails else 'NOT READY'}: {fails} problem(s), {warns} thing(s) to look at")
    print("Before driving: press ORIGIN with the car on its start mark.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
