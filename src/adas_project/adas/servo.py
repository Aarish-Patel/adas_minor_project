"""Steering command -> the two servo angles (the same Ackermann maths as rc_controller.py).

Kept here so the Pi runtime and the simulator's viewer use one implementation. The
numbers are the servo calibration; they live in tuning.json under "servo".
"""

import math
from dataclasses import dataclass


@dataclass
class ServoCalibration:
    left_right: float = 30.0
    left_center: float = 90.0
    left_left: float = 140.0
    right_right: float = 30.0
    right_center: float = 90.0
    right_left: float = 140.0
    left_scale: float = 1.0           # servo degrees per wheel degree
    right_scale: float = 1.0
    left_channel: int = 2             # ESP32 output the left servo is wired to (1 = GPIO18, 2 = GPIO19)
    right_channel: int = 1
    ackermann_factor: float = 1.0     # 1 = ideal, 0 = parallel wheels, <0 = anti-Ackermann
    steering_expo: float = 1.0
    motor_reversed: bool = True       # forward needs a negative PWM on this car (matches rc_controller.py)
    servo_sign: int = -1              # ADAS uses + = physically left. On this car the servo maths treats
                                      # + as physically right (rc_controller STEERING_SIGN = 1), hence -1.
                                      # Flip it if the car steers the wrong way on the first drive.


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def steering_to_servo_angles(steer, cal, wheelbase=0.20, track=0.07):
    """steer -1..+1 (+ = left) -> (left_servo_deg, right_servo_deg)."""
    steer = _clamp(steer, -1.0, 1.0)
    turning_left = steer > 0
    max_inner = (cal.left_left - cal.left_center) if turning_left else (cal.right_center - cal.right_right)
    inner = (abs(steer) ** cal.steering_expo) * max_inner
    if inner < 0.01:
        outer = 0.0
    else:
        ideal = math.degrees(math.atan(1.0 / (1.0 / math.tan(math.radians(inner)) + track / wheelbase)))
        outer = inner - cal.ackermann_factor * (inner - ideal)
    if turning_left:
        left_wheel, right_wheel, sign = inner, outer, 1
    else:
        left_wheel, right_wheel, sign = outer, inner, -1
    left = cal.left_center + sign * left_wheel * cal.left_scale
    right = cal.right_center + sign * right_wheel * cal.right_scale
    return (_clamp(left, cal.left_right, cal.left_left), _clamp(right, cal.right_right, cal.right_left))


def command_text(steer, pwm, cal, wheelbase=0.20, track=0.07):
    """The two lines the ESP32 firmware understands: servo angles and motor PWM.

    steer is the ADAS convention (+ = physically left), pwm + = forward.
    """
    left, right = steering_to_servo_angles(cal.servo_sign * steer, cal, wheelbase, track)
    by_channel = {cal.left_channel: left, cal.right_channel: right}
    motor = -pwm if cal.motor_reversed else pwm
    return f"A {by_channel[1]:.1f} {by_channel[2]:.1f}\nM {int(round(motor))}\n"
