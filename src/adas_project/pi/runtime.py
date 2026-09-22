"""The on-car control loop. Hardware is plugged in through four small objects:

    lidar.poll()      -> None or (t, angles_rad_ccw, ranges_m, valid)     newest scan, once
    camera.poll()     -> None or (t, [MarkerObs])                        newest frame's markers, once
    driver.read()     -> (steer, pwm, buttons)                           steer -1..1 (+ left), pwm -255..255
    actuator.send(steer, pwm)                                            to the ESP32

The very same Runtime class is exercised against the simulator in pi/hil.py, so what is
tested there is what runs on the Pi. Only the four small adapters differ.
"""

import time

from adas.lidar_utils import scan_to_points


class Runtime:
    def __init__(self, tuning, lidar, camera, driver, actuator, logger=None, mode="active"):
        self.tuning = tuning
        self.pipeline = tuning.make_pipeline()
        self.lidar, self.camera, self.driver, self.actuator = lidar, camera, driver, actuator
        self.logger = logger
        self.mode = mode
        self.t = 0.0
        self.last = None
        self.status = {}
        self._buttons_prev = {}

    def _edge(self, buttons, name):
        now = bool(buttons.get(name))
        was = self._buttons_prev.get(name, False)
        self._buttons_prev[name] = now
        return now and not was

    def step(self, t):
        """One control tick at time t (seconds). Returns (steer_sent, pwm_sent, level)."""
        dt = 0.02 if self.last is None else max(1e-3, t - self.last)
        self.last = t
        self.t = t
        p = self.pipeline

        scan = self.lidar.poll() if self.lidar else None
        if scan is not None:
            ts, angles, ranges, valid = scan
            p.on_scan(scan_to_points(angles, ranges, valid, self.tuning.vehicle), ts)
        frame = self.camera.poll() if self.camera else None
        if frame is not None:
            p.on_markers(frame[1], frame[0])

        steer, pwm, buttons = self.driver.read()
        if self._edge(buttons, "mode"):
            self.mode = {"off": "advisory", "advisory": "active", "active": "off"}[self.mode]
        if self._edge(buttons, "park"):
            p.parking.stop() if p.parking.active else p.parking.start()
        if self._edge(buttons, "follow"):
            p.acc.enabled = not p.acc.enabled
        if buttons.get("estop"):
            steer, pwm = 0.0, 0.0

        pwm_out, level, info = p.on_control(dt, steer, pwm, self.mode)
        steer_out = p.steer_out
        self.actuator.send(steer_out, pwm_out)

        self.status = {"level": level, "mode": self.mode, "D": info.get("D"), "park": p.parking.state}
        if self.logger:
            self.logger.log(t, steer, pwm, p)
        return steer_out, pwm_out, level

    def run(self, hz=50.0, stop=lambda: False):
        period = 1.0 / hz
        start = time.perf_counter()
        nxt = start
        try:
            while not stop():
                now = time.perf_counter()
                if now >= nxt:
                    self.step(now - start)
                    nxt += period
                else:
                    time.sleep(min(period / 4, nxt - now))
        finally:
            self.actuator.send(0.0, 0.0)
