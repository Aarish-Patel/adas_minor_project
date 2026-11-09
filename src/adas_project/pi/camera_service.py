"""The relay keeps the marker camera service (pi/marker_camera.py) running (TODO AP2): starts it, restarts it when it dies, stops it with the relay.

No systemd unit and no sudo: the service is a child process of the relay (it dies with it: PR_SET_PDEATHSIG where available) and only starts when a
webcam device exists. `pi/camera.json` "autostart": false turns it off (the webcam draws power: on a Pi with a weak supply that matters).
"""
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESTART_AFTER_S = 8.0           # a service that exits is started again after this long (a missing webcam must not be retried in a tight loop)


def _die_with_parent():
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)          # PR_SET_PDEATHSIG
    except Exception:
        pass


class CameraService:
    def __init__(self, cfg, cmd=None, log_path=None, device_check=True):
        self.cfg = cfg
        self.cmd = cmd or [sys.executable, "-u", os.path.join(HERE, "marker_camera.py")]
        self.log_path = log_path or os.path.expanduser("~/marker_camera.log")
        self.device_check, self.proc, self.starts, self.running = device_check, None, 0, True
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera-service")

    def device_present(self):
        return os.path.exists(f"/dev/video{int(self.cfg.get('index', 0))}")

    def start(self):
        if not self.cfg.get("autostart", True):
            return False
        self.thread.start()
        return True

    def _run(self):
        while self.running:
            if self.proc is None or self.proc.poll() is not None:
                if self.proc is not None:
                    time.sleep(RESTART_AFTER_S)                     # it exited (webcam unplugged, crash): do not spin
                    if not self.running:
                        return
                if not self.device_check or self.device_present():
                    try:
                        with open(self.log_path, "ab") as log:          # (the child keeps its own copy of the handle)
                            self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                         preexec_fn=_die_with_parent if os.name == "posix" else None, close_fds=True)
                        self.starts += 1
                    except OSError:
                        self.proc = None
                else:
                    time.sleep(RESTART_AFTER_S)
            time.sleep(1.0)

    def stop(self):
        self.running = False
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
