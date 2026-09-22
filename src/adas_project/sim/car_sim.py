"""Virtual car: Ackermann kinematics plus simple motor and steering dynamics."""

import math
from dataclasses import dataclass

from adas.aeb import SpeedModel
from adas.vehicle_params import steer_to_delta


@dataclass
class Dynamics:
    """The 'real' car. Differences from the ADAS's assumed model test its robustness."""
    speed_model: SpeedModel = SpeedModel()
    tau_motor: float = 0.15        # s, speed response
    accel_max: float = 3.0         # m/s^2
    brake_max: float = 2.5         # m/s^2 when the motor is driven against the motion
    coast_decel: float = 1.5       # m/s^2 with the motor off
    tau_steer: float = 0.05        # s, servo response


class SimCar:
    def __init__(self, x, y, theta, params, dyn):
        self.p = params
        self.dyn = dyn
        self.x, self.y, self.theta = x, y, theta
        self.v = 0.0
        self.steer = 0.0           # actual steering, -1..+1
        self.collided = False
        self.impact_speed = 0.0
        self.clearance = math.inf

    def step(self, dt, steer_cmd, pwm_cmd, world):
        if self.collided:
            self.v = 0.0
            return

        d = self.dyn
        self.steer += (steer_cmd - self.steer) * min(1.0, dt / d.tau_steer)

        target = d.speed_model.speed(pwm_cmd)
        accel = (target - self.v) / d.tau_motor
        lower = -d.coast_decel if target == 0 else -d.brake_max
        accel = max(lower, min(d.accel_max, accel))
        new_v = self.v + accel * dt
        if target == 0 and self.v * new_v < 0:
            new_v = 0.0
        self.v = new_v

        delta = steer_to_delta(self.steer, self.p)
        omega = self.v * math.tan(delta) / self.p.wheelbase
        mid = self.theta + 0.5 * omega * dt
        self.x += self.v * math.cos(mid) * dt
        self.y += self.v * math.sin(mid) * dt
        self.theta += omega * dt

        self.clearance = world.clearance(self.x, self.y, self.theta, self.p)
        if self.clearance <= 0.0:
            self.collided = True
            self.impact_speed = abs(self.v)
            self.v = 0.0
