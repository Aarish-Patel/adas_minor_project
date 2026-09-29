"""The ESP32 serial link with health supervision (TODO O1).

On 28 Sep the ESP32 went silent and later dropped off USB while the relay kept writing motor commands into the void:
nothing told the driver. This wrapper:
  - pings the ESP32 once a second from a background thread and reads its replies (no blocking read in the control
    loop any more); no PONG for SILENT_AFTER s -> state "silent" (powered but not running: dead chip, brown-out loop)
  - notices the firmware's boot banner ("ROVER READY") -> the ESP32 rebooted, typically a brown-out when the motor or
    servos draw current
  - on a write/read error (USB unplugged, re-enumerated) closes the port, keeps looking for the ESP32 - first by its
    stable /dev/serial/by-id name, else any /dev/ttyUSB* that is not the LiDAR - and reopens it
  - lets the relay run without an ESP32 at all (LiDAR, GUI and logging keep working; motor commands are dropped)
States: "ok", "silent", "lost" (no port), "connecting" (just opened, no reply yet).
"""
import glob
import os
import threading
import time


class EspLink:
    PING_EVERY = 1.0
    SILENT_AFTER = 3.0
    REOPEN_EVERY = 1.0

    def __init__(self, open_fn, port, lidar_port=None, on_event=None, candidates=None):
        """open_fn(port) -> an opened serial object (pyserial, or the simulator's stand-in); candidates() -> ports
        to try when reopening (default: the by-id name, else every ttyUSB that is not the LiDAR)."""
        self.open_fn, self.port, self.lidar_port = open_fn, port, lidar_port
        if candidates is not None:
            self._candidates = candidates
        self.on_event = on_event or (lambda msg, **kw: None)
        self.by_id = self._by_id(port)
        self.ser = None
        self.lock = threading.Lock()
        self.state, self.detail = "lost", "not opened yet"
        self.last_pong = self.last_rx = self.last_ping = self.opened_at = 0.0
        self.reboots, self.reconnects, self.dropped = 0, 0, 0
        self.last_line = ""
        self.running = True
        if port:
            self._open(port)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._supervisor, daemon=True).start()

    # ---------------------------------------------------------------- serial-like surface used by the relay
    def write(self, data):
        with self.lock:
            ser = self.ser
            if ser is None:
                self.dropped += 1
                return 0
            try:
                return ser.write(data)
            except Exception as e:                 # SerialException / OSError: the port went away
                self._lost_locked(e)
                return 0

    def close(self):
        self.running = False
        with self.lock:
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    def healthy(self):
        return self.state == "ok"

    def status(self):
        now = time.time()
        return {"state": self.state, "detail": self.detail, "port": self.port,
                "pong_age_s": round(now - self.last_pong, 1) if self.last_pong else None,
                "reboots": self.reboots, "reconnects": self.reconnects, "dropped_writes": self.dropped}

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _by_id(port):
        if not port or not port.startswith("/dev/"):
            return None
        for link in glob.glob("/dev/serial/by-id/*"):
            if os.path.realpath(link) == os.path.realpath(port):
                return link
        return None

    def _candidates(self):
        if self.by_id and os.path.exists(self.by_id):
            return [self.by_id]
        lidar = os.path.realpath(self.lidar_port) if self.lidar_port and self.lidar_port.startswith("/dev/") else None
        return [p for p in sorted(glob.glob("/dev/ttyUSB*")) if os.path.realpath(p) != lidar]

    def _open(self, port):
        try:
            ser = self.open_fn(port)
        except Exception as e:
            self.detail = f"open {port} failed: {e}"
            return False
        with self.lock:
            self.ser, self.port = ser, port
            self.opened_at = time.time()
            self.state, self.detail = "connecting", f"opened {port}"
        return True

    def _lost_locked(self, err):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        if self.state != "lost":
            self.state, self.detail = "lost", f"{type(err).__name__}: {err}"
            self.on_event("ESP32 link lost", error=str(err))
            print(f"\nESP32 LINK LOST ({err}) - motor commands dropped, looking for it again", flush=True)

    def _reader(self):
        buf = b""
        while self.running:
            ser = self.ser
            if ser is None:
                time.sleep(0.1)
                continue
            try:
                n = ser.in_waiting
                data = ser.read(n or 1)
            except Exception as e:
                with self.lock:
                    if self.ser is ser:
                        self._lost_locked(e)
                continue
            if not data:
                time.sleep(0.01)
                continue
            now = time.time()
            self.last_rx = now
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode(errors="replace").strip()
                if not text:
                    continue
                if "PONG" in text:
                    self.last_pong = now
                else:
                    self.last_line = text
                    if "ROVER READY" in text:
                        self.reboots += 1
                        self.on_event("ESP32 rebooted (boot banner) - brown-out?", count=self.reboots)
                        print(f"\nESP32 REBOOTED (#{self.reboots}) - power dip? check the motor/servo supply", flush=True)
            if len(buf) > 512:
                buf = b""

    def _supervisor(self):
        last_try = 0.0
        while self.running:
            time.sleep(0.25)
            now = time.time()
            if self.ser is None:
                if now - last_try >= self.REOPEN_EVERY:
                    last_try = now
                    for p in self._candidates():
                        if self._open(p):
                            self.reconnects += 1
                            self.on_event("ESP32 link reopened", port=p)
                            print(f"\nESP32 link reopened on {p}", flush=True)
                            break
                continue
            if now - self.last_ping >= self.PING_EVERY:
                self.last_ping = now
                self.write(b"PING\n")
            fresh = self.last_pong and now - self.last_pong < self.SILENT_AFTER
            if fresh and self.state != "ok":
                was = self.state
                self.state, self.detail = "ok", f"answering on {self.port}"
                if was == "silent":
                    self.on_event("ESP32 answering again")
            elif not fresh and now - self.opened_at > self.SILENT_AFTER and self.state in ("ok", "connecting"):
                self.state = "silent"
                self.detail = f"no reply to PING for {self.SILENT_AFTER:.0f} s on {self.port} (dead chip or reset loop?)"
                self.on_event("ESP32 silent", port=self.port)
                print(f"\nESP32 SILENT: {self.detail}", flush=True)
