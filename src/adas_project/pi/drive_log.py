"""Drive logger: records everything needed to rebuild the car's behaviour offline.

One session = one file  /home/pi/logs/<YYYYmmdd_HHMMSS>_<source>.jsonl.gz, one JSON record per line:

  {"k":"meta", "t":..., "source":..., "tuning":{...}, "git":...}          first line
  {"k":"cmd",  "t":..., "line":"M -120"}                                   every line sent to the ESP32
  {"k":"scan", "t":..., "a":[raw angle*100...], "d":[mm...], "q":[quality...]}
                RAW sensor angles (before the yaw offset), so a later yaw calibration can be re-applied
  {"k":"drv",  "t":..., ...}                                               driver input / ADAS decision (relay)
  {"k":"ev",   "t":..., "msg":...}                                         events: stops, state changes, errors

Scans are kept at full rate while the motor is commanded (and for 2 s after), 1 per second while parked.
Writing happens on a background thread, so the scan and control loops never wait for the SD card.
Disable with RC_NO_LOG=1.
"""
import gzip
import json
import os
import queue
import subprocess
import threading
import time

LOG_DIR = os.environ.get("RC_LOG_DIR", "/home/pi/logs")
IDLE_SCAN_PERIOD_S = 1.0
ACTIVE_HOLD_S = 2.0


class DriveLog:
    def __init__(self, source, tuning_path=None, log_dir=LOG_DIR, enabled=None):
        self.enabled = (os.environ.get("RC_NO_LOG") != "1") if enabled is None else enabled
        self.path = None
        self._q = queue.Queue(maxsize=5000)
        self._last_active = 0.0
        self._last_idle_scan = 0.0
        self.dropped = 0
        if not self.enabled:
            return
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S") + f"_{source}.jsonl.gz")
        tuning = None
        if tuning_path and os.path.exists(tuning_path):
            try:
                tuning = json.load(open(tuning_path))
            except Exception:
                pass
        try:
            git = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=3).stdout.strip() or None
        except Exception:
            git = None
        self._put({"k": "meta", "t": time.time(), "source": source, "tuning": tuning, "git": git, "version": 1})
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- recording
    def _put(self, rec):
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            self.dropped += 1            # never block a control loop; count what was lost instead

    def cmd(self, line):
        if not self.enabled:
            return
        now = time.time()
        line = line.strip()
        if line.startswith("M ") and line != "M 0":
            self._last_active = now
        self._put({"k": "cmd", "t": now, "line": line})

    def scan(self, raw, t=None):
        """raw: iterable of (quality, raw_angle_deg, dist_mm) exactly as the RPLIDAR library yields them."""
        if not self.enabled:
            return
        now = time.time() if t is None else t
        if now - self._last_active > ACTIVE_HOLD_S:           # parked: thin to one scan per second
            if now - self._last_idle_scan < IDLE_SCAN_PERIOD_S:
                return
            self._last_idle_scan = now
        a, d, q = [], [], []
        for qual, ang, dist in raw:
            if dist > 0:
                a.append(int(round(ang * 100)))
                d.append(int(round(dist)))
                q.append(int(qual))
        self._put({"k": "scan", "t": now, "a": a, "d": d, "q": q})

    def driver(self, **fields):
        if self.enabled:
            self._put({"k": "drv", "t": time.time(), **fields})

    def event(self, msg, **fields):
        if self.enabled:
            self._put({"k": "ev", "t": time.time(), "msg": msg, **fields})

    # ---------------------------------------------------------------- writer
    def _writer(self):
        with gzip.open(self.path, "wt", compresslevel=3) as f:
            last_flush = time.time()
            while True:
                try:
                    rec = self._q.get(timeout=0.5)
                except queue.Empty:
                    rec = None
                if rec is not None:
                    if rec.get("k") == "_close":
                        break
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                if time.time() - last_flush > 1.0:
                    f.flush()                # a crash loses at most about a second
                    last_flush = time.time()

    def close(self):
        if self.enabled and self.path:
            if self.dropped:
                self.event("records dropped (queue full)", n=self.dropped)
            self._put({"k": "_close"})
            self._thread.join(timeout=5)


class LoggedSerial:
    """Wraps the ESP32 serial port: every line written is recorded, then sent unchanged."""

    def __init__(self, ser, log):
        self._ser, self._log = ser, log

    def write(self, data):
        for line in data.decode(errors="replace").splitlines():
            if line.strip():
                self._log.cmd(line)
        return self._ser.write(data)

    def __getattr__(self, name):
        return getattr(self._ser, name)


def _records(path):
    """Yield records; stops cleanly at a truncated line or a missing gzip end marker (a power cut, or a log
    that is still being written - everything flushed so far is readable)."""
    with gzip.open(path, "rt") as f:
        try:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    return
        except (EOFError, OSError):
            return


def load(path):
    """Read a log back: returns dict(meta, cmds=[(t, line)], scans=[(t, angle_deg[], dist_m[], quality[])], drv, ev)."""
    out = {"meta": None, "cmds": [], "scans": [], "drv": [], "ev": []}
    for r in _records(path):
        k = r.get("k")
        if k == "meta":
            out["meta"] = r
        elif k == "cmd":
            out["cmds"].append((r["t"], r["line"]))
        elif k == "scan":
            out["scans"].append((r["t"], [x / 100.0 for x in r["a"]], [x / 1000.0 for x in r["d"]], r["q"]))
        elif k == "drv":
            out["drv"].append(r)
        elif k == "ev":
            out["ev"].append(r)
    return out
