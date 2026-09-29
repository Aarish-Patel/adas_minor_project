"""Run the whole calibration + demo sequence, one step after another (started from the control panel).

Between groups it PAUSES and tells you (on the panel) how to set up the arena; press Continue there.
Every step is a normal script run with RC_NO_HOLD=1 (no results-screen wait). A step fails if it exits with an
error, times out, or - for calibrations - does not produce a fresh result. The run stops at the first failure,
so a bad calibration never feeds the next test. Nothing is applied automatically: the summary lists each new
value next to the current one and you press Apply.
"""
import json
import os
import signal
import subprocess
import sys
import time

ROOT = "/home/pi/rc_car"
PI = ROOT + "/pi"
STATE = PI + "/run_all_state.json"
CONTINUE = PI + "/run_all_continue.flag"
RESULTS = PI + "/cal_results.json"
REPORT = PI + "/run_all_report.json"

# (kind, title, script, result key, timeout s, prompt)
STEPS = [
    ("wait", "Set up: LiDAR front", None, None, 0,
     "Put ONE object dead-centre in front of the car, 0.3-1.2 m away. Then press Continue."),
    ("run", "Calibrate LiDAR front", "pi/lidar_front_cal.py", "lidar", 90, ""),
    ("wait", "Set up: open area", None, None, 0,
     "Remove the object. The car moves next: about 1.6 m clear ahead, 0.5 m behind, free space to both sides. Then press Continue."),
    ("run", "Calibrate steering centre", "pi/center_fine.py", "servo_center", 900, ""),
    ("run", "Calibrate speed + stopping", "pi/brake_run.py", "speed", 600, ""),
    ("run", "Calibrate turning", "pi/turn_test.py", "turn", 700, ""),
    ("wait", "Set up: obstacle avoidance", None, None, 0,
     "Put ONE obstacle straight ahead about 1.4 m from the car, with about 3.5 m clear along the line and room on both sides. Then press Continue."),
    ("run", "Obstacle avoidance demo", "pi/bypass_run.py", None, 120, ""),
]

state = {"step": 0, "total": len(STEPS), "title": "", "status": "starting", "prompt": "", "waiting": False, "log": [], "results": []}
child = None


def save():
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    for attempt in range(20):          # Windows refuses the rename while a reader has the file open
        try:
            os.replace(tmp, STATE)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)


def note(line):
    state["log"] = (state["log"] + [time.strftime("%H:%M:%S ") + line])[-30:]
    save()
    print(line, flush=True)


def wait_for_continue(prompt, poll=0.5, state_path=None):
    if os.path.exists(CONTINUE):
        os.remove(CONTINUE)
    state.update(waiting=True, prompt=prompt, status="waiting for you")
    save()
    while not os.path.exists(CONTINUE):
        time.sleep(poll)
    os.remove(CONTINUE)
    state.update(waiting=False, prompt="", status="running")
    save()


def fresh_result(key, since):
    try:
        r = json.load(open(RESULTS)).get(key)
    except Exception:
        return None
    if r and r.get("time", "") >= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since)):
        return r
    return None


def run_step(title, script, key, timeout, cmd=None):
    """Run one script. Returns (ok, message, result)."""
    global child
    since = time.time() - 1
    env = dict(os.environ, RC_NO_HOLD="1", PYTHONUNBUFFERED="1")
    child = subprocess.Popen(cmd or [sys.executable, "-u", ROOT + "/" + script], cwd=ROOT, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    tail = []
    import threading

    def pump():
        for line in child.stdout:
            tail.append(line.rstrip())
            del tail[:-6]
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    while child.poll() is None:
        if time.time() - t0 > timeout:
            child.terminate()
            try:
                child.wait(8)
            except subprocess.TimeoutExpired:
                child.kill()
            return False, f"timed out after {timeout} s", None
        time.sleep(0.3)
    th.join(2)
    if child.returncode != 0:
        return False, f"exited with code {child.returncode}: {' | '.join(tail[-3:])}", None
    if key:
        r = fresh_result(key, since)
        if r is None:
            return False, "finished but produced no valid result (see the log)", None
        return True, "ok", r
    return True, "ok", None


def run_steps(steps=STEPS):
    results = []
    for i, (kind, title, script, key, timeout, prompt) in enumerate(steps):
        state.update(step=i + 1, total=len(steps), title=title)
        if kind == "wait":
            note(f"[{i + 1}/{len(steps)}] {title}: waiting for you")
            wait_for_continue(prompt)
            continue
        state.update(status="running", waiting=False, prompt="")
        note(f"[{i + 1}/{len(steps)}] {title} ...")
        ok, msg, res = run_step(title, script, key, timeout)
        results.append({"step": title, "ok": ok, "message": msg, "result": res})
        state["results"] = results
        note(f"    -> {'OK' if ok else 'FAILED'}: {msg}")
        if not ok:
            state.update(status=f"stopped: {title} failed", waiting=False)
            save()
            with open(REPORT, "w") as f:
                json.dump(results, f, indent=1)
            return False
    state.update(status="all steps finished - review the results and press Apply", waiting=False, title="done")
    save()
    with open(REPORT, "w") as f:
        json.dump(results, f, indent=1)
    return True


def _term(*_):
    if child is not None and child.poll() is None:
        child.terminate()
    state.update(status="stopped", waiting=False)
    save()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _term)
    sys.exit(0 if run_steps() else 1)
