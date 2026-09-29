"""High-density LiDAR scans through Slamtec's SDK (pi/lidar_stream/rc_lidar_stream), with the same methods
the code already uses on the `rplidar` library, so either can be used:

    lidar = open_lidar("/dev/ttyUSB1")
    for scan in lidar.iter_scans():          # each scan: [(quality, angle_deg, dist_mm), ...], one rotation
        ...

On this car's A3 the SDK's default mode gives ~960 valid points per rotation vs ~210 from the `rplidar`
library's standard scan (measured on the Pi). Set RC_LIDAR_DENSE=0 to force the old library.
"""
import os
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
STREAM_BIN = os.path.join(HERE, "lidar_stream", "rc_lidar_stream")


class DenseLidar:
    def __init__(self, port, baudrate=256000, timeout=3):
        self.port, self.baudrate, self.timeout = port, baudrate, timeout
        self.mode = None
        self._proc = None
        self._err = []

    # ---- rplidar-compatible surface
    def get_info(self):
        self._ensure()
        return {"driver": "slamtec sdk", "mode": self.mode}

    def get_health(self):
        return ("Good", 0)

    def iter_scans(self, max_buf_meas=None, min_len=5):
        self._ensure()
        for line in self._proc.stdout:
            p = line.split()
            if len(p) < 2 or p[0] == "MODE":
                continue
            v = p[2:]
            scan = [(int(v[i + 2]), int(v[i]) * 90.0 / 16384.0, int(v[i + 1]) / 4.0) for i in range(0, len(v) - 2, 3)]
            if len(scan) >= min_len:
                yield scan
        raise RuntimeError("LiDAR stream ended: " + " | ".join(self._err[-3:]))

    def stop(self):
        self._shutdown()

    def stop_motor(self):
        self._shutdown()

    def disconnect(self):
        self._shutdown()

    def clean_input(self):
        pass

    # ---- process handling
    def _ensure(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen([STREAM_BIN, self.port, str(self.baudrate)], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._pump_err, daemon=True).start()
        first = self._proc.stdout.readline().split()
        if not first or first[0] != "MODE":
            self._shutdown()
            raise RuntimeError("LiDAR stream did not start: " + " ".join(first) + " " + " | ".join(self._err[-3:]))
        self.mode = first[1] if len(first) > 1 else "?"

    def _pump_err(self):
        p = self._proc
        for line in p.stderr:
            self._err.append(line.strip())
            del self._err[:-20]

    def _shutdown(self):
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            p.terminate()                     # the helper stops the scan and the motor on SIGTERM
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()


def dense_available():
    return os.environ.get("RC_LIDAR_DENSE", "1") != "0" and os.access(STREAM_BIN, os.X_OK)


def open_lidar(port, baudrate=256000, timeout=3):
    """High-density stream if the helper is built, otherwise the `rplidar` library (standard scan)."""
    if dense_available():
        return DenseLidar(port, baudrate, timeout)
    from rplidar import RPLidar
    return RPLidar(port, baudrate=baudrate, timeout=timeout)
