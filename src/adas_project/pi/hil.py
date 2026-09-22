"""Simulator-in-the-loop: the real Pi Runtime driving the simulated car.

    python -m pi.hil wall        emergency stop
    python -m pi.hil parking     auto-park (button pressed at the start)
    python -m pi.hil signs       traffic signs

The Simulator plays the physical world only (car, LiDAR, camera). The ADAS decisions come
from the Runtime, exactly the code that runs on the Pi. Commands also go through the real
protocol code (servo mapping + UDP) into an emulated ESP32, and the emulator's servo angles are
compared with what the servo mapping should give.
"""

import math
import sys

from adas.config import Tuning
from adas.servo import steering_to_servo_angles
from pi.link import Esp32Actuator, Esp32Emulator, UdpLink
from pi.runtime import Runtime
from sim.library import SCENARIOS
from sim.simulator import Simulator


class SimLidar:
    def __init__(self, sim):
        self.sim, self._last = sim, -1

    def poll(self):
        if self.sim.scan_id != self._last and self.sim.last_scan is not None:
            self._last = self.sim.scan_id
            return self.sim.last_scan
        return None


class SimCamera:
    def __init__(self, sim):
        self.sim, self._last = sim, 0

    def poll(self):
        if self.sim.marker_serial != self._last:
            self._last = self.sim.marker_serial
            return self.sim.t, self.sim.last_markers
        return None


class ScriptedDriver:
    def __init__(self, fn):
        self.fn, self.t = fn, 0.0

    def read(self):
        return self.fn(self.t)


class RecordingActuator:
    """Sends over UDP to the emulator, and remembers the last command for the simulator."""

    def __init__(self, inner):
        self.inner, self.steer, self.pwm = inner, 0.0, 0.0

    def send(self, steer, pwm):
        self.steer, self.pwm = steer, pwm
        self.inner.send(steer, pwm)


def run(scenario, seconds=30.0, verbose=True):
    world, start = SCENARIOS[scenario]["build"]()
    sim = Simulator(world, start, adas_on="off")             # the simulator is only the physical world here
    tuning = Tuning()

    emu = Esp32Emulator()
    emu.start()
    actuator = RecordingActuator(Esp32Actuator(UdpLink("127.0.0.1", emu.port), tuning.servo))

    script = {"wall": lambda t: (0.0, 220.0 if t > 0.3 else 0.0, {}),
              "parking": lambda t: (0.0, 0.0, {"park": t < 0.2 and t > 0.05}),
              "signs": lambda t: (0.0, 200.0, {}),
              "pedestrian": lambda t: (0.0, 200.0 if t > 0.3 else 0.0, {}),
              "leader": lambda t: (0.0, 220.0 if t > 0.3 else 0.0, {"follow": t < 0.2 and t > 0.05})}[scenario]
    driver = ScriptedDriver(script)
    rt = Runtime(tuning, SimLidar(sim), SimCamera(sim), driver, actuator)

    next_ctrl, worst_servo_err = 0.0, 0.0
    recent = []
    steer_cmd, pwm_cmd = 0.0, 0.0
    while sim.t < seconds and not sim.car.collided:
        if sim.t >= next_ctrl:
            driver.t = sim.t
            steer_cmd, pwm_cmd, _ = rt.step(sim.t)
            next_ctrl += 0.02
            left, right = steering_to_servo_angles(tuning.servo.servo_sign * steer_cmd, tuning.servo)
            exp = {tuning.servo.left_channel: left, tuning.servo.right_channel: right}
            recent = (recent + [(exp[1], exp[2])])[-6:]
            # UDP is asynchronous: the emulator must match one of the last few commands sent
            if emu.servo1 is not None and len(recent) >= 6:
                err = min(max(abs(emu.servo1 - a), abs(emu.servo2 - b)) for a, b in recent)
                worst_servo_err = max(worst_servo_err, err)
        sim.step(steer_cmd, pwm_cmd)
        if scenario == "parking" and rt.pipeline.parking.state in ("done", "failed"):
            break
        if scenario in ("wall",) and sim.t > 6:
            break
    emu.stop()

    result = {"scenario": scenario, "t": round(sim.t, 1), "crashed": bool(sim.car.collided),
              "min_gap_cm": round(sim.min_clearance * 100, 1), "commands_received_by_esp32": emu.commands,
              "servo_protocol_max_error_deg": round(worst_servo_err, 2), "park_state": rt.pipeline.parking.state,
              "x": round(sim.car.x, 2)}
    if verbose:
        print(result)
    return result


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "wall")
